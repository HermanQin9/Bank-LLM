package com.bankfraud.api;

import com.bankfraud.model.TransactionAlertRecord;
import com.bankfraud.repository.TransactionAlertRepository;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URI;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.Executors;

/**
 * Minimal HTTP server that exposes transaction_alerts JSON payloads to
 * dashboards.
 */
public class TransactionAlertRestServer {

    private static final Logger logger = LoggerFactory.getLogger(TransactionAlertRestServer.class);
    private static final String ALERTS_PATH = "/api/alerts";

    private final TransactionAlertRepository repository;
    private final HttpServer server;
    private final ObjectMapper objectMapper;

    public TransactionAlertRestServer(int port) throws IOException {
        this.repository = new TransactionAlertRepository();
        this.server = HttpServer.create(new InetSocketAddress(port), 0);
        this.objectMapper = new ObjectMapper();
        this.objectMapper.registerModule(new JavaTimeModule());
        this.objectMapper.disable(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS);
        this.server.createContext(ALERTS_PATH, new AlertHandler());
        this.server.setExecutor(Executors.newFixedThreadPool(4));
    }

    public void start() {
        server.start();
        logger.info("Transaction alert REST server running on http://localhost:{}{}", server.getAddress().getPort(),
                ALERTS_PATH);
    }

    public void stop() {
        server.stop(0);
        logger.info("Transaction alert REST server stopped");
    }

    private class AlertHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            try {
                if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                    sendResponse(exchange, 405, Map.of("error", "Method not allowed"));
                    return;
                }

                URI uri = exchange.getRequestURI();
                String path = uri.getPath();
                if (ALERTS_PATH.equals(path) || (ALERTS_PATH + "/").equals(path)) {
                    handleList(exchange, uri.getQuery());
                } else if (path.startsWith(ALERTS_PATH + "/")) {
                    handleDetail(exchange, path.substring((ALERTS_PATH + "/").length()));
                } else {
                    sendResponse(exchange, 404, Map.of("error", "Not found"));
                }
            } catch (Exception e) {
                logger.error("Failed to handle request", e);
                sendResponse(exchange, 500, Map.of("error", "Internal server error"));
            }
        }

        private void handleList(HttpExchange exchange, String query) throws IOException {
            int limit = resolveLimit(query);
            List<TransactionAlertRecord> alerts = repository.findRecent(limit);
            Map<String, Object> payload = new HashMap<>();
            payload.put("count", alerts.size());
            payload.put("data", alerts);
            sendResponse(exchange, 200, payload);
        }

        private void handleDetail(HttpExchange exchange, String alertId) throws IOException {
            if (alertId == null || alertId.isBlank()) {
                sendResponse(exchange, 400, Map.of("error", "Alert id missing"));
                return;
            }
            Optional<TransactionAlertRecord> alert = repository.findByIdWithEvidence(alertId);
            if (alert.isEmpty()) {
                sendResponse(exchange, 404, Map.of("error", "Alert not found"));
                return;
            }
            sendResponse(exchange, 200, Map.of("data", alert.get()));
        }

        private int resolveLimit(String query) {
            if (query == null || query.isBlank()) {
                return 25;
            }
            for (String token : query.split("&")) {
                String[] parts = token.split("=");
                if (parts.length == 2 && "limit".equalsIgnoreCase(parts[0])) {
                    try {
                        return Math.max(1, Math.min(Integer.parseInt(parts[1]), 200));
                    } catch (NumberFormatException ignore) {
                        return 25;
                    }
                }
            }
            return 25;
        }

        private void sendResponse(HttpExchange exchange, int status, Object body) throws IOException {
            byte[] json = objectMapper.writeValueAsBytes(body);
            Headers headers = exchange.getResponseHeaders();
            headers.add("Content-Type", "application/json; charset=utf-8");
            exchange.sendResponseHeaders(status, json.length);
            try (OutputStream os = exchange.getResponseBody()) {
                os.write(json);
            }
        }
    }

    public static void main(String[] args) throws IOException {
        int port = Integer.parseInt(System.getenv().getOrDefault("ALERT_API_PORT", "8085"));
        TransactionAlertRestServer server = new TransactionAlertRestServer(port);
        server.start();
        Runtime.getRuntime().addShutdownHook(new Thread(server::stop));
    }
}
