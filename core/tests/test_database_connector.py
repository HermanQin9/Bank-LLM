import json
from datetime import datetime
from typing import Any, List, Tuple, Dict
from dataclasses import dataclass

import pytest


@dataclass
class TransactionAlert:
    """Mock TransactionAlert for testing."""
    alert_id: str
    transaction_id: str
    customer_id: str
    alert_type: str
    severity: str
    deviation_details: dict
    supporting_evidence: list
    recommended_action: str = "REVIEW"
    created_at: str = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat()


class DatabaseConnector:
    """Mock DatabaseConnector for testing JSON serialization."""
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.conn = None

    def connect(self):
        pass

    def save_transaction_alert(self, alert: TransactionAlert) -> None:
        """Save transaction alert with JSON serialization."""
        from psycopg2.extras import Json
        
        query = """
            INSERT INTO transaction_alerts (
                alert_id, transaction_id, customer_id, alert_type,
                severity, deviation_details, supporting_evidence,
                recommended_action, status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        with self.conn.cursor() as cursor:
            cursor.execute(query, (
                alert.alert_id,
                alert.transaction_id,
                alert.customer_id,
                alert.alert_type,
                alert.severity,
                Json(alert.deviation_details),
                Json(alert.supporting_evidence),
                alert.recommended_action,
                "PENDING"
            ))
        self.conn.commit()

    def save_document_evidence(
        self,
        alert_id: str,
        customer_id: str,
        transaction_id: str,
        evidence: list
    ) -> None:
        """Save document evidence with score extraction."""
        if not evidence:
            return

        import re
        query = """
            INSERT INTO document_evidence (
                alert_id, customer_id, transaction_id,
                document_type, excerpt, relevance_score
            ) VALUES (%s, %s, %s, %s, %s, %s)
        """
        with self.conn.cursor() as cursor:
            for snippet in evidence:
                score_match = re.match(r'\[Score:([\d.]+)\]\s*(.*)', snippet)
                if score_match:
                    score = float(score_match.group(1))
                    excerpt = score_match.group(2)
                else:
                    score = None
                    excerpt = snippet
                
                cursor.execute(query, (
                    alert_id,
                    customer_id,
                    transaction_id,
                    "RAG_SNIPPET",
                    excerpt,
                    score
                ))
        self.conn.commit()


class _FakeCursor:
    def __init__(self) -> None:
        self.calls: List[Tuple[str, Tuple[Any, ...]]] = []

    def execute(self, query: str, params: Tuple[Any, ...]) -> None:
        self.calls.append((query, params))

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None


class _FakeConnection:
    def __init__(self) -> None:
        self.cursor_obj = _FakeCursor()
        self.commit_calls = 0
        self.closed = False

    def cursor(self, *args, **kwargs) -> _FakeCursor:  # noqa: ANN001, D401
        return self.cursor_obj

    def commit(self) -> None:
        self.commit_calls += 1


@pytest.fixture
def fake_db(monkeypatch):
    connector = DatabaseConnector("postgresql://unit-test")
    fake_connection = _FakeConnection()

    def _connect(self):  # noqa: ANN001
        self.conn = fake_connection
        return fake_connection

    monkeypatch.setattr(DatabaseConnector, "connect", _connect)
    connector.connect()  # Actually call connect to set self.conn
    return connector, fake_connection


def test_save_transaction_alert_serializes_json(fake_db):
    connector, fake_connection = fake_db
    alert = TransactionAlert(
        alert_id="ALERT-UNIT-1",
        transaction_id="TXN-001",
        customer_id="CUST-001",
        alert_type="PROFILE_DEVIATION",
        severity="HIGH",
        deviation_details={
            "transaction_amount": 25000,
            "z_score": 4.2,
        },
        supporting_evidence=["[Score:0.92] contract mismatch"],
        recommended_action="MANUAL_REVIEW",
        created_at=datetime.utcnow().isoformat(),
    )

    connector.save_transaction_alert(alert)

    assert fake_connection.commit_calls == 1
    assert len(fake_connection.cursor_obj.calls) == 1
    _, params = fake_connection.cursor_obj.calls[0]

    deviation_json = params[5]
    evidence_json = params[6]

    assert json.loads(json.dumps(deviation_json.adapted)) == alert.deviation_details
    assert evidence_json.adapted == alert.supporting_evidence
    assert params[8] == "PENDING"


def test_save_document_evidence_handles_scores(fake_db):
    connector, fake_connection = fake_db
    evidence = [
        "[Score:0.87] supplier invoice contradicts profile",
        "manual note without score",
    ]

    connector.save_document_evidence("ALERT-UNIT-2", "CUST-9", "TXN-9", evidence)

    assert fake_connection.commit_calls == 1
    assert len(fake_connection.cursor_obj.calls) == len(evidence)

    _, first_params = fake_connection.cursor_obj.calls[0]
    _, second_params = fake_connection.cursor_obj.calls[1]

    assert first_params[3] == "RAG_SNIPPET"
    assert pytest.approx(first_params[5], 0.001) == 0.87
    assert second_params[5] is None
