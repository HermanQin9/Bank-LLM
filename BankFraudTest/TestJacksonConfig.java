import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import java.util.HashMap;
import java.util.Map;

public class TestJacksonConfig {
    public static void main(String[] args) throws Exception {
        ObjectMapper mapper = new ObjectMapper();
        mapper.setPropertyNamingStrategy(PropertyNamingStrategies.SNAKE_CASE);

        Map<String, Object> data = new HashMap<>();
        data.put("transactionId", "TEST-001");
        data.put("customerId", "CUST_001");
        data.put("amount", 5000.0);
        data.put("merchantName", "Amazon");

        String json = mapper.writeValueAsString(data);
        System.out.println("Generated JSON: " + json);

        // Expected:
        // {"transaction_id":"TEST-001","customer_id":"CUST_001","amount":5000.0,"merchant_name":"Amazon"}
    }
}
