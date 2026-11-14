#include "nanobot_network.h"

#include <pthread.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;

static void expect(int condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        failures++;
    }
}

static char message_not_found[256];
static char message_invalid[256];
static volatile int race_failures = 0;

/* Every thread fails one call, then copies the diagnostic under the library
 * lock. The copy must always be one complete known message. */
static void *race_errors(void *arg) {
    const int which = *(const int *)arg;
    char scratch[8];
    char message[256];
    for (int i = 0; i < 4000; i++) {
        if (which == 0) {
            (void)nanobot_network_status(999, scratch, (int32_t)sizeof(scratch));
        } else {
            (void)nanobot_network_start("{}");
        }
        int32_t length = nanobot_network_last_error_copy(message, (int32_t)sizeof(message));
        if (length <= 0 || length >= (int32_t)sizeof(message)) {
            race_failures++;
            return NULL;
        }
        if (message[length] != '\0') {
            race_failures++;
            return NULL;
        }
        if (strcmp(message, message_not_found) != 0 && strcmp(message, message_invalid) != 0) {
            race_failures++;
        }
    }
    return NULL;
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

    (void)nanobot_network_status(999, buffer, sizeof(buffer));
    expect(nanobot_network_last_error_copy(message_not_found, (int32_t)sizeof(message_not_found)) > 0,
           "copy of the handle diagnostic");
    (void)nanobot_network_start("{}");
    expect(nanobot_network_last_error_copy(message_invalid, (int32_t)sizeof(message_invalid)) > 0,
           "copy of the config diagnostic");
    expect(strcmp(message_not_found, message_invalid) != 0, "diagnostics are distinct");
    expect(nanobot_network_last_error() != NULL, "legacy diagnostic pointer");

    pthread_t threads[4];
    int kinds[4] = {0, 1, 0, 1};
    for (int i = 0; i < 4; i++) {
        expect(pthread_create(&threads[i], NULL, race_errors, &kinds[i]) == 0, "start diagnostic thread");
    }
    for (int i = 0; i < 4; i++) {
        (void)pthread_join(threads[i], NULL);
    }
    expect(race_failures == 0, "concurrent diagnostics stay whole and identified");

    if (failures != 0) {
        return 1;
    }
    printf("NANOBOT_CABI_SMOKE_OK version=%d\n", nanobot_network_abi_version());
    return 0;
}
