import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;

public class QuickURLConnectionTest {
    public static void main(String[] args) throws Exception {
        URL url = new URL("http://localhost:8000/api/analyze-transaction");
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", "application/json; charset=utf-8");
        conn.setDoOutput(true);

        String json = "{" +
                "\"transaction_id\":\"HTTPURLCONN\"," +
                "\"customer_id\":\"CUST_001\"," +
                "\"amount\":5000," +
                "\"merchant_name\":\"Amazon\"" +
                "}";

        try (OutputStream os = conn.getOutputStream()) {
            os.write(json.getBytes());
        }

        System.out.println("Status: " + conn.getResponseCode());
        try (java.io.InputStream is = conn.getInputStream()) {
            String response = new String(is.readAllBytes());
            System.out.println(response);
        }
    }
}
