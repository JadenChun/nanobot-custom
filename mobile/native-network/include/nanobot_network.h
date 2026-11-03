#ifndef NANOBOT_NETWORK_H
#define NANOBOT_NETWORK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Nanobot mobile native transport ABI, version 1.
 *
 * Inputs are NUL-terminated UTF-8 JSON except for WebSocket frame bytes.
 * The caller owns every input buffer. The library copies input data before
 * returning. Returned C strings (last_error) are owned by the library and must
 * not be freed by the caller; they remain valid until the next error or stop.
 *
 * Error codes:
 *    0 ok
 *   -1 invalid request
 *   -2 denied by policy
 *   -3 operation already in progress
 *   -4 handle not found
 *   -5 bounded queue full
 *   -6 payload too large
 *   -7 output buffer too small
 *   -8 transport unavailable
 *   -9 transport closed
 *  -10 login URL is not approved
 * -100 internal error
 *
 * No function accepts a raw URL, dial target, LocalAPI request or arbitrary
 * filesystem path. HTTP operations are named and resolved against an exact
 * confirmed origin and an allowlisted method/path table.
 */

#define NANOBOT_NETWORK_ABI_VERSION 1

int32_t nanobot_network_abi_version(void);

/* config_json is a Config object. Returns a positive handle or 0 with error. */
int64_t nanobot_network_start(const char *config_json);

/* Writes a status JSON object. Returns bytes written or a negative error. */
int32_t nanobot_network_status(int64_t handle, char *out, int32_t capacity);

/* Writes {"login_url":"..."} when an approved URL is available. */
int32_t nanobot_network_login_url(int64_t handle, char *out, int32_t capacity);

/* Starts a named HTTP request asynchronously. Returns a request id. */
int32_t nanobot_network_http(int64_t handle, const char *request_json, uint64_t *out_request_id);

/* Cancels a request. A result or cancellation event may still be queued. */
int32_t nanobot_network_cancel(int64_t handle, uint64_t request_id);

/* Opens the single app-owned WebSocket. Returns ws id 1. */
int32_t nanobot_network_ws_open(int64_t handle, const char *request_json, uint64_t *out_ws_id);

/* kind 0 = text, 1 = binary. The library copies data. */
int32_t nanobot_network_ws_send(int64_t handle, uint64_t ws_id, int32_t kind,
                                const uint8_t *data, int32_t length);

int32_t nanobot_network_ws_close(int64_t handle, uint64_t ws_id);

/* Sends one protocol ping and waits for its pong for at most five seconds. */
int32_t nanobot_network_ws_ping(int64_t handle, uint64_t ws_id);

/*
 * Copies one JSON event into out. Returns 0 when empty, the event length when
 * copied, -7 when the buffer is too small (the event is retained), or another
 * negative error. A 512 KiB buffer handles ordinary events; the JNI bridge
 * retries with 12 MiB when a media response event is larger.
 */
int32_t nanobot_network_poll_event(int64_t handle, char *out, int32_t capacity);

int32_t nanobot_network_stop(int64_t handle);

const char *nanobot_network_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* NANOBOT_NETWORK_H */
