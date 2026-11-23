/* pthread barriers need the POSIX feature set under -std=c11. */
#define _POSIX_C_SOURCE 200809L

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

/* Requirement case C2 (independent requirements U2/R2): two threads run one
 * failing native operation each and immediately copy the diagnostic into a
 * private buffer. Every iteration must return that thread's own error text. */
#define C2_ROUNDS 2000

static pthread_barrier_t c2_start;
static pthread_barrier_t c2_nc_first;
static pthread_barrier_t c2_nc_second;
static volatile int c2_failures = 0;

struct c2_worker {
    int which;
    const char *expect;
};

static void *c2_identity(void *arg) {
    const struct c2_worker *worker = arg;
    char scratch[8];
    char message[256];
    (void)pthread_barrier_wait(&c2_start);
    for (int i = 0; i < C2_ROUNDS; i++) {
        if (worker->which == 0) {
            (void)nanobot_network_status(999, scratch, (int32_t)sizeof(scratch));
        } else {
            (void)nanobot_network_start("{}");
        }
        int32_t length = nanobot_network_last_error_copy(message, (int32_t)sizeof(message));
        if (length <= 0 || length != (int32_t)strlen(worker->expect) ||
            message[length] != '\0') {
            if (c2_failures < 3) {
                fprintf(stderr, "C2 cross-identity: own=%s got=%s len=%d\n",
                        worker->expect, message, (int)length);
            }
            c2_failures++;
            return NULL;
        }
        if (strcmp(message, worker->expect) != 0) {
            if (c2_failures < 3) {
                fprintf(stderr, "C2 cross-identity: own=%s got=%s\n", worker->expect, message);
            }
            c2_failures++;
            return NULL;
        }
    }
    return NULL;
}

/* Negative control: A fails and copies, then B fails again, then A copies once
 * more. The second copy must still be A's own text, never B's. Both threads
 * reach every barrier, so a failure cannot hang the run. */
static void *c2_negative_a(void *arg) {
    const struct c2_worker *worker = arg;
    char scratch[8];
    char message[256];
    (void)nanobot_network_status(999, scratch, (int32_t)sizeof(scratch));
    int32_t first = nanobot_network_last_error_copy(message, (int32_t)sizeof(message));
    (void)pthread_barrier_wait(&c2_nc_first);
    (void)pthread_barrier_wait(&c2_nc_second);
    char again[256];
    int32_t second = nanobot_network_last_error_copy(again, (int32_t)sizeof(again));
    if (first <= 0 || strcmp(message, worker->expect) != 0 ||
        second <= 0 || strcmp(again, worker->expect) != 0) {
        fprintf(stderr, "C2 negative control: own=%s first=%s second=%s\n",
                worker->expect, message, again);
        c2_failures++;
    }
    return NULL;
}

static void *c2_negative_b(void *arg) {
    (void)arg;
    (void)pthread_barrier_wait(&c2_nc_first);
    (void)nanobot_network_start("{}");
    (void)pthread_barrier_wait(&c2_nc_second);
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

    /* C2: each concurrent caller must read its own error text. */
    (void)pthread_barrier_init(&c2_start, NULL, 2);
    (void)pthread_barrier_init(&c2_nc_first, NULL, 2);
    (void)pthread_barrier_init(&c2_nc_second, NULL, 2);
    struct c2_worker worker_a = {0, message_not_found};
    struct c2_worker worker_b = {1, message_invalid};
    pthread_t c2_threads[4];
    expect(pthread_create(&c2_threads[0], NULL, c2_identity, &worker_a) == 0, "start C2 thread A");
    expect(pthread_create(&c2_threads[1], NULL, c2_identity, &worker_b) == 0, "start C2 thread B");
    for (int i = 0; i < 2; i++) {
        (void)pthread_join(c2_threads[i], NULL);
    }
    expect(c2_failures == 0, "C2 concurrent callers keep their own error text");
    expect(pthread_create(&c2_threads[2], NULL, c2_negative_a, &worker_a) == 0, "start C2 negative A");
    expect(pthread_create(&c2_threads[3], NULL, c2_negative_b, &worker_b) == 0, "start C2 negative B");
    for (int i = 2; i < 4; i++) {
        (void)pthread_join(c2_threads[i], NULL);
    }
    expect(c2_failures == 0, "C2 negative control keeps thread A's text");
    (void)pthread_barrier_destroy(&c2_start);
    (void)pthread_barrier_destroy(&c2_nc_first);
    (void)pthread_barrier_destroy(&c2_nc_second);

    if (failures != 0) {
        return 1;
    }
    printf("NANOBOT_CABI_SMOKE_OK version=%d\n", nanobot_network_abi_version());
    return 0;
}
