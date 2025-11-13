package com.bankfraud.streaming;

import com.bankfraud.model.FraudAlert;
import com.bankfraud.model.Transaction;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.ProducerConfig;
import org.apache.kafka.clients.producer.ProducerRecord;
import org.apache.kafka.common.serialization.StringSerializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Closeable;
import java.time.Duration;
import java.util.Properties;
import java.util.concurrent.Future;

/**
 * Kafka publisher that broadcasts fraud alerts to downstream AI services.
 */
public class TransactionKafkaPublisher implements Closeable {

    private static final Logger logger = LoggerFactory.getLogger(TransactionKafkaPublisher.class);

    private final KafkaProducer<String, String> producer;
    private final String topic;
    private final ObjectMapper objectMapper;
    private final boolean enabled;

    public TransactionKafkaPublisher(String bootstrapServers, String topic, String clientId) {
        this.topic = topic;
        this.objectMapper = new ObjectMapper();
        this.enabled = bootstrapServers != null && !bootstrapServers.isBlank();

        if (!enabled) {
            this.producer = null;
            logger.warn("Kafka publisher disabled - no bootstrap servers configured");
            return;
        }

        Properties props = new Properties();
        props.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, bootstrapServers);
        props.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());
        props.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());
        props.put(ProducerConfig.ACKS_CONFIG, "all");
        props.put(ProducerConfig.ENABLE_IDEMPOTENCE_CONFIG, true);
        props.put(ProducerConfig.RETRIES_CONFIG, 3);
        props.put(ProducerConfig.LINGER_MS_CONFIG, 10);
        props.put(ProducerConfig.BATCH_SIZE_CONFIG, 32_768);
        props.put(ProducerConfig.COMPRESSION_TYPE_CONFIG, "zstd");
        props.put(ProducerConfig.CLIENT_ID_CONFIG, clientId != null ? clientId : "fraud-alert-producer");

        this.producer = new KafkaProducer<>(props);
        Runtime.getRuntime().addShutdownHook(new Thread(this::close));
        logger.info("Kafka publisher initialized for topic '{}'", topic);
    }

    public boolean isEnabled() {
        return enabled;
    }

    public void publishAlert(Transaction transaction, FraudAlert alert, double ruleScore, double finalScore) {
        if (!enabled) {
            return;
        }
        try {
            TransactionAlertEvent event = TransactionAlertEvent.from(transaction, alert, ruleScore, finalScore);
            String payload = objectMapper.writeValueAsString(event);
            ProducerRecord<String, String> record = new ProducerRecord<>(
                    topic,
                    transaction.getCustomerId(),
                    payload);

            Future<?> metadata = producer.send(record, (md, ex) -> {
                if (ex != null) {
                    logger.error("Failed to publish alert {} to Kafka", event.getAlertId(), ex);
                } else if (logger.isDebugEnabled()) {
                    logger.debug("Alert {} written to {}-{}@{}", event.getAlertId(), md.topic(), md.partition(),
                            md.offset());
                }
            });
            metadata.get();
        } catch (JsonProcessingException e) {
            logger.error("Unable to serialize alert for Kafka", e);
        } catch (Exception e) {
            logger.error("Kafka publish failed", e);
        }
    }

    @Override
    public void close() {
        if (producer != null) {
            logger.info("Closing Kafka producer for topic '{}'", topic);
            producer.flush();
            producer.close(Duration.ofSeconds(5));
        }
    }
}
