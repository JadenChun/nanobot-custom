#include "nanobot_network.h"

#include <stdio.h>
#include <string.h>

static int failures = 0;

static void expect(int condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        failures++;
    }
}

int main(void) {
    char buffer[512];
    expect(nanobot_network_abi_version() == NANOBOT_NETWORK_ABI_VERSION, "ABI version");
    expect(nanobot_network_start("{}") == 0, "invalid config must not create a handle");
    expect(strstr(nanobot_network_last_error(), "origin") != NULL, "last error names the invalid config");
    expect(nanobot_network_start("{\"origin\":\"http://evil.example\",\"installation_id\":\"i\",\"hostname\":\"h\",\"state_dir\":\"/tmp/nanobot-cabi\"}") == 0,
           "plain HTTP origin must be rejected");
    expect(strstr(nanobot_network_last_error(), "https") != NULL, "last error names the HTTPS rule");
    expect(nanobot_network_status(999, buffer, sizeof(buffer)) == -4, "unknown status handle");
    expect(nanobot_network_login_url(999, buffer, sizeof(buffer)) == -4, "unknown login handle");
    expect(nanobot_network_poll_event(999, buffer, sizeof(buffer)) == -4, "unknown poll handle");
    expect(nanobot_network_pending_event_size(999) == -4, "unknown event-size handle");
    expect(nanobot_network_stop(999) == -4, "unknown stop handle");
    expect(nanobot_network_cancel(999, 1) == -4, "unknown cancel handle");
    uint64_t request_id = 0;
    expect(nanobot_network_http(999, "{}", &request_id) == -4, "unknown HTTP handle");
    expect(nanobot_network_ws_open(999, "{}", &request_id) == -4, "unknown WS handle");
    expect(nanobot_network_ws_send(999, 1, 0, (const uint8_t *)"x", 1) == -4, "unknown WS send handle");
    expect(nanobot_network_ws_close(999, 1) == -4, "unknown WS close handle");

    if (failures != 0) {
        return 1;
    }
    printf("NANOBOT_CABI_SMOKE_OK version=%d\n", nanobot_network_abi_version());
    return 0;
}
