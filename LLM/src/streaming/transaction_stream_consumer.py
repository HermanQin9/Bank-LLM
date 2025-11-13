"""Kafka consumer that streams Java fraud alerts into the LLM workflow."""

from __future__ import annotations

import json
import os
import signal
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import psycopg2
import psycopg2.extras
from confluent_kafka import Consumer, KafkaException
from loguru import logger

from src.llm_engine.langgraph_agent import DocumentProcessingAgent, LANGGRAPH_AVAILABLE
from src.llm_engine.universal_client import UniversalLLMClient
from src.rag_system.gemini_rag_pipeline import GeminiRAGPipeline


def _env(key: str, default: str) -> str:
    value = os.getenv(key)
    return value if value not in (None, "") else default


@dataclass
class KafkaSettings:
    bootstrap_servers: str
    topic: str = "fraud.alerts"
    group_id: str = "fraud-alert-consumers"


@dataclass
class DatabaseSettings:
    host: str
    port: int
    database: str
    user: str
    password: str

    @property
    def dsn(self) -> str:
        return (
            f"host={self.host} port={self.port} dbname={self.database} "
            f"user={self.user} password={self.password}"
        )


class TransactionAlertStreamConsumer:
    """Consumes Kafka events and enriches them with LLM + RAG context."""

    def __init__(
        self,
        kafka: KafkaSettings,
        database: DatabaseSettings,
        llm_provider: str = "auto",
    ) -> None:
        self.config = kafka
        self.db_config = database
        self.consumer = Consumer(
            {
                "bootstrap.servers": kafka.bootstrap_servers,
                "group.id": kafka.group_id,
                "auto.offset.reset": "earliest",
                "enable.auto.commit": False,
            }
        )
        self.llm_provider = llm_provider
        self.llm_client = UniversalLLMClient(provider=llm_provider)
        self.document_agent: Optional[DocumentProcessingAgent] = (
            DocumentProcessingAgent(llm_provider=llm_provider) if LANGGRAPH_AVAILABLE else None
        )
        self.rag_pipeline = GeminiRAGPipeline(llm_provider=llm_provider)
        self.running = False

    def start(self) -> None:
        logger.info(
            "Starting Kafka consumer on topic '{}' (group={})",
            self.config.topic,
            self.config.group_id,
        )
        self.running = True
        self.consumer.subscribe([self.config.topic])
        while self.running:
            try:
                message = self.consumer.poll(1.0)
                if message is None:
                    continue
                if message.error():
                    raise KafkaException(message.error())
                payload = json.loads(message.value())
                self._process_event(payload)
                self.consumer.commit(message)
            except KeyboardInterrupt:
                logger.info("Stopping consumer (keyboard interrupt)")
                self.stop()
            except KafkaException as exc:
                logger.error("Kafka error: {}", exc)
                time.sleep(5)
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("Failed to process stream message: {}", exc)

    def stop(self) -> None:
        self.running = False
        self.consumer.close()

    def _process_event(self, event: Dict[str, Any]) -> None:
        logger.info(
            "Processing alert {} for customer {} (risk={})",
            event.get("alertId"),
            event.get("customerId"),
            event.get("riskLevel"),
        )
        docs = self._retrieve_supporting_documents(event)
        summary = self._generate_summary(event, docs)
        self._persist_document_evidence(event, summary, docs)

    def _retrieve_supporting_documents(self, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        query = (
            f"customer {event.get('customerId')} transaction {event.get('transactionId')} "
            f"risk {event.get('riskLevel')}"
        )
        return self.rag_pipeline.semantic_search_only(query=query, top_k=3)

    def _generate_summary(
        self,
        event: Dict[str, Any],
        documents: List[Dict[str, Any]],
    ) -> str:
        context_parts = [
            f"Transaction {event.get('transactionId')} with merchant {event.get('merchantName')} "
            f"amount {event.get('amount')}"
        ]
        for idx, doc in enumerate(documents, start=1):
            context_parts.append(f"[Doc {idx}] {doc.get('document', '')[:400]}")
        document_text = "\n\n".join(context_parts)

        if self.document_agent is not None:
            result = self.document_agent.process_document(document_text)
            return result.get("analysis") or result.get("workflow_messages", [""])[-1]

        prompt = (
            "You are a fraud investigation copilot. Given the transaction summary and supporting "
            "documents, produce a concise risk narrative (<=200 words) highlighting why the "
            "transaction was flagged and what actions to take.\n\n" + document_text
        )
        return self.llm_client.generate(prompt, temperature=0.3)

    def _persist_document_evidence(
        self,
        event: Dict[str, Any],
        summary: str,
        documents: List[Dict[str, Any]],
    ) -> None:
        key_entities = {
            "merchant": event.get("merchantName"),
            "risk_level": event.get("riskLevel"),
            "rules": event.get("rulesTriggered"),
        }
        risk_indicators = {
            "kafka_event_id": event.get("eventId"),
            "topic": self.config.topic,
        }
        if documents:
            risk_indicators["vector_store_hits"] = len(documents)

        insert_sql = """
            INSERT INTO document_evidence (
                alert_id,
                transaction_id,
                customer_id,
                document_type,
                document_path,
                excerpt,
                relevance_score
            ) VALUES (%s,%s,%s,%s,%s,%s,%s)
        """

        with psycopg2.connect(self.db_config.dsn) as conn:
            with conn.cursor() as cur:
                final_score = event.get("finalScore")
                try:
                    relevance = float(final_score) if final_score is not None else 0.0
                except (TypeError, ValueError):
                    relevance = 0.0
                cur.execute(
                    insert_sql,
                    (
                        event.get("alertId"),
                        event.get("transactionId"),
                        event.get("customerId"),
                        "AUTO_SUMMARY",
                        f"kafka://{self.config.topic}",
                        summary,
                        relevance,
                    ),
                )
                self._upsert_transaction_alert(conn, event, summary, documents, risk_indicators)
        logger.info(
            "Stored document evidence for alert {} and synced transaction_alerts",
            event.get("alertId"),
        )

    def _upsert_transaction_alert(
        self,
        conn,
        event: Dict[str, Any],
        summary: str,
        documents: List[Dict[str, Any]],
        risk_meta: Dict[str, Any],
    ) -> None:
        alert_id = event.get("alertId") or f"ALERT-{event.get('transactionId')}"
        supporting_payload: List[str] = [summary]
        for doc in documents:
            snippet = doc.get("document", "")
            score = doc.get("score")
            prefix = f"[Score:{score:.3f}] " if isinstance(score, (int, float)) else ""
            supporting_payload.append(f"{prefix}{snippet[:400]}")

        deviation_details = {
            "rule_score": event.get("ruleBasedScore"),
            "final_score": event.get("finalScore"),
            "detection_method": event.get("detectionMethod"),
            "merchant": event.get("merchantName"),
            "amount": event.get("amount"),
            "currency": event.get("currency"),
            "risk_meta": risk_meta,
        }

        sql = """
            INSERT INTO transaction_alerts (
                alert_id,
                transaction_id,
                customer_id,
                alert_type,
                severity,
                deviation_details,
                supporting_evidence,
                recommended_action,
                status
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (alert_id) DO UPDATE SET
                severity = EXCLUDED.severity,
                deviation_details = EXCLUDED.deviation_details,
                supporting_evidence = EXCLUDED.supporting_evidence,
                recommended_action = EXCLUDED.recommended_action,
                status = EXCLUDED.status,
                resolved_at = CASE WHEN EXCLUDED.status = 'CLOSED' THEN NOW() ELSE transaction_alerts.resolved_at END
        """

        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    alert_id,
                    event.get("transactionId"),
                    event.get("customerId"),
                    event.get("detectionMethod", "STREAMING_ALERT"),
                    event.get("riskLevel", "UNKNOWN"),
                    psycopg2.extras.Json(deviation_details),
                    psycopg2.extras.Json(supporting_payload),
                    self._recommended_action(event.get("riskLevel")),
                    "PENDING",
                ),
            )

    @staticmethod
    def _recommended_action(risk_level: Optional[str]) -> str:
        mapping = {
            "CRITICAL": "BLOCK_TRANSACTION",
            "HIGH": "MANUAL_REVIEW",
            "MEDIUM": "FLAG_FOR_MONITORING",
        }
        return mapping.get((risk_level or "").upper(), "REVIEW")

    @classmethod
    def from_env(cls) -> "TransactionAlertStreamConsumer":
        kafka = KafkaSettings(
            bootstrap_servers=_env("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            topic=_env("KAFKA_ALERT_TOPIC", "fraud.alerts"),
            group_id=_env("KAFKA_GROUP_ID", "fraud-alert-consumers"),
        )
        database = DatabaseSettings(
            host=_env("POSTGRES_HOST", "localhost"),
            port=int(_env("POSTGRES_PORT", "5432")),
            database=_env("POSTGRES_DB", "frauddb"),
            user=_env("POSTGRES_USER", "postgres"),
            password=_env("POSTGRES_PASSWORD", "postgres"),
        )
        provider = _env("LLM_PROVIDER", "auto")
        return cls(kafka=kafka, database=database, llm_provider=provider)


def _install_signal_handlers(consumer: TransactionAlertStreamConsumer) -> None:
    def _handler(signum: int, _frame: Any) -> None:  # noqa: ARG001
        logger.info("Received signal %s, stopping consumer", signum)
        consumer.stop()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


if __name__ == "__main__":
    stream_consumer = TransactionAlertStreamConsumer.from_env()
    _install_signal_handlers(stream_consumer)
    stream_consumer.start()
