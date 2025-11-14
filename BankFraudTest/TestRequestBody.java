import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import java.util.HashMap;
import java.util.Map;

public class TestRequestBody {
    public static void main(String[] args) throws Exception {
        ObjectMapper mapper = new ObjectMapper();
        mapper.setPropertyNamingStrategy(PropertyNamingStrategies.SNAKE_CASE);

        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("transaction_id", "TEST-001");
        requestBody.put("customer_id", "CUST_001");
        requestBody.put("amount", 5000.0);
        requestBody.put("merchant_name", "Amazon");

        String json = mapper.writeValueAsString(requestBody);
        System.out.println(json);
    }
}
