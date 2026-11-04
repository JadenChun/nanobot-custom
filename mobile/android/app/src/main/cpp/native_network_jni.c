#include <jni.h>
#include <stdint.h>
#include <stdlib.h>

#include "nanobot_network.h"

static jbyteArray copy_bytes(JNIEnv *env, const char *data, int32_t length) {
    if (length <= 0) {
        return NULL;
    }
    jbyteArray result = (*env)->NewByteArray(env, length);
    if (result != NULL) {
        (*env)->SetByteArrayRegion(env, result, 0, length, (const jbyte *)data);
    }
    return result;
}

JNIEXPORT jint JNICALL
Java_ai_nanobot_mobile_network_NativeNetwork_abiVersion(JNIEnv *env, jobject thiz) {
    (void)env;
    (void)thiz;
    return (jint)nanobot_network_abi_version();
}

JNIEXPORT jlong JNICALL
Java_ai_nanobot_mobile_network_NativeNetwork_start(JNIEnv *env, jobject thiz, jstring config) {
    (void)thiz;
    const char *raw = (*env)->GetStringUTFChars(env, config, NULL);
    if (raw == NULL) {
        return 0;
    }
    int64_t handle = nanobot_network_start(raw);
    (*env)->ReleaseStringUTFChars(env, config, raw);
    return (jlong)handle;
}

JNIEXPORT jbyteArray JNICALL
Java_ai_nanobot_mobile_network_NativeNetwork_status(JNIEnv *env, jobject thiz, jlong handle) {
    (void)thiz;
    char *buffer = malloc(64 * 1024);
    if (buffer == NULL) {
        return NULL;
    }
    int32_t length = nanobot_network_status((int64_t)handle, buffer, 64 * 1024);
    jbyteArray result = copy_bytes(env, buffer, length);
    free(buffer);
    return result;
}

JNIEXPORT jbyteArray JNICALL
Java_ai_nanobot_mobile_network_NativeNetwork_loginUrl(JNIEnv *env, jobject thiz, jlong handle) {
    (void)thiz;
    char *buffer = malloc(16 * 1024);
    if (buffer == NULL) {
        return NULL;
    }
    int32_t length = nanobot_network_login_url((int64_t)handle, buffer, 16 * 1024);
    jbyteArray result = copy_bytes(env, buffer, length);
    free(buffer);
    return result;
}

JNIEXPORT jlong JNICALL
Java_ai_nanobot_mobile_network_NativeNetwork_http(JNIEnv *env, jobject thiz, jlong handle,
                                                  jstring request) {
    (void)thiz;
    const char *raw = (*env)->GetStringUTFChars(env, request, NULL);
    if (raw == NULL) {
        return 0;
    }
    uint64_t request_id = 0;
    int32_t status = nanobot_network_http((int64_t)handle, raw, &request_id);
    (*env)->ReleaseStringUTFChars(env, request, raw);
    return status == 0 ? (jlong)request_id : 0;
}

JNIEXPORT jint JNICALL
Java_ai_nanobot_mobile_network_NativeNetwork_cancel(JNIEnv *env, jobject thiz, jlong handle,
                                                    jlong request_id) {
    (void)env;
    (void)thiz;
    return (jint)nanobot_network_cancel((int64_t)handle, (uint64_t)request_id);
}

JNIEXPORT jlong JNICALL
Java_ai_nanobot_mobile_network_NativeNetwork_wsOpen(JNIEnv *env, jobject thiz, jlong handle,
                                                    jstring request) {
    (void)thiz;
    const char *raw = (*env)->GetStringUTFChars(env, request, NULL);
    if (raw == NULL) {
        return 0;
    }
    uint64_t ws_id = 0;
    int32_t status = nanobot_network_ws_open((int64_t)handle, raw, &ws_id);
    (*env)->ReleaseStringUTFChars(env, request, raw);
    return status == 0 ? (jlong)ws_id : 0;
}

JNIEXPORT jint JNICALL
Java_ai_nanobot_mobile_network_NativeNetwork_wsSend(JNIEnv *env, jobject thiz, jlong handle,
                                                    jlong ws_id, jint kind, jbyteArray data) {
    (void)thiz;
    jsize length = (*env)->GetArrayLength(env, data);
    jbyte *bytes = NULL;
    if (length > 0) {
        bytes = (*env)->GetByteArrayElements(env, data, NULL);
        if (bytes == NULL) {
            return -1;
        }
    }
    int32_t status = nanobot_network_ws_send((int64_t)handle, (uint64_t)ws_id, (int32_t)kind,
                                             (const uint8_t *)bytes, (int32_t)length);
    if (bytes != NULL) {
        (*env)->ReleaseByteArrayElements(env, data, bytes, JNI_ABORT);
    }
    return (jint)status;
}

JNIEXPORT jint JNICALL
Java_ai_nanobot_mobile_network_NativeNetwork_wsClose(JNIEnv *env, jobject thiz, jlong handle,
                                                     jlong ws_id) {
    (void)env;
    (void)thiz;
    return (jint)nanobot_network_ws_close((int64_t)handle, (uint64_t)ws_id);
}

JNIEXPORT jint JNICALL
Java_ai_nanobot_mobile_network_NativeNetwork_wsPing(JNIEnv *env, jobject thiz, jlong handle,
                                                    jlong ws_id) {
    (void)env;
    (void)thiz;
    return (jint)nanobot_network_ws_ping((int64_t)handle, (uint64_t)ws_id);
}

JNIEXPORT jbyteArray JNICALL
Java_ai_nanobot_mobile_network_NativeNetwork_pollEvent(JNIEnv *env, jobject thiz, jlong handle) {
    (void)thiz;
    int32_t capacity = 512 * 1024;
    char *buffer = malloc(capacity);
    if (buffer == NULL) {
        return NULL;
    }
    int32_t length = nanobot_network_poll_event((int64_t)handle, buffer, capacity);
    if (length == -7) {
        free(buffer);
        capacity = 12 * 1024 * 1024;
        buffer = malloc(capacity);
        if (buffer == NULL) {
            return NULL;
        }
        length = nanobot_network_poll_event((int64_t)handle, buffer, capacity);
    }
    if (length < 0) {
        length = 0;
    }
    jbyteArray result = copy_bytes(env, buffer, length);
    free(buffer);
    return result;
}

JNIEXPORT jint JNICALL
Java_ai_nanobot_mobile_network_NativeNetwork_stop(JNIEnv *env, jobject thiz, jlong handle) {
    (void)env;
    (void)thiz;
    return (jint)nanobot_network_stop((int64_t)handle);
}

JNIEXPORT jstring JNICALL
Java_ai_nanobot_mobile_network_NativeNetwork_lastError(JNIEnv *env, jobject thiz) {
    (void)thiz;
    const char *message = nanobot_network_last_error();
    return (*env)->NewStringUTF(env, message == NULL ? "" : message);
}

JNIEXPORT jint JNICALL
Java_ai_nanobot_mobile_network_NativeNetwork_mediaSetToken(JNIEnv *env, jobject thiz,
                                                          jlong handle, jstring token) {
    (void)thiz;
    const char *raw = (*env)->GetStringUTFChars(env, token, NULL);
    if (raw == NULL) {
        return -1;
    }
    int32_t status = nanobot_network_media_set_token((int64_t)handle, (char *)raw);
    (*env)->ReleaseStringUTFChars(env, token, raw);
    return status;
}

JNIEXPORT jlong JNICALL
Java_ai_nanobot_mobile_network_NativeNetwork_mediaOpen(JNIEnv *env, jobject thiz, jlong handle,
                                                       jstring path, jstring range_header) {
    (void)thiz;
    const char *raw_path = (*env)->GetStringUTFChars(env, path, NULL);
    if (raw_path == NULL) {
        return 0;
    }
    const char *raw_range = (*env)->GetStringUTFChars(env, range_header, NULL);
    if (raw_range == NULL) {
        (*env)->ReleaseStringUTFChars(env, path, raw_path);
        return 0;
    }
    uint64_t stream_id = 0;
    int32_t status = nanobot_network_media_open((int64_t)handle, (char *)raw_path,
                                                (char *)raw_range, &stream_id);
    (*env)->ReleaseStringUTFChars(env, path, raw_path);
    (*env)->ReleaseStringUTFChars(env, range_header, raw_range);
    return status == 0 ? (jlong)stream_id : 0;
}

JNIEXPORT jbyteArray JNICALL
Java_ai_nanobot_mobile_network_NativeNetwork_mediaInfo(JNIEnv *env, jobject thiz, jlong handle,
                                                       jlong stream_id) {
    (void)thiz;
    char *buffer = malloc(16 * 1024);
    if (buffer == NULL) {
        return NULL;
    }
    int32_t length = nanobot_network_media_info((int64_t)handle, (uint64_t)stream_id, buffer,
                                                16 * 1024);
    jbyteArray result = copy_bytes(env, buffer, length);
    free(buffer);
    return result;
}

JNIEXPORT jint JNICALL
Java_ai_nanobot_mobile_network_NativeNetwork_mediaRead(JNIEnv *env, jobject thiz, jlong handle,
                                                       jlong stream_id, jbyteArray buffer) {
    (void)thiz;
    if (buffer == NULL) {
        return -1;
    }
    jsize capacity = (*env)->GetArrayLength(env, buffer);
    jbyte *bytes = (*env)->GetByteArrayElements(env, buffer, NULL);
    if (bytes == NULL) {
        return -1;
    }
    int32_t read = nanobot_network_media_read((int64_t)handle, (uint64_t)stream_id,
                                              (char *)bytes, capacity);
    (*env)->ReleaseByteArrayElements(env, buffer, bytes, 0);
    return (jint)read;
}

JNIEXPORT jint JNICALL
Java_ai_nanobot_mobile_network_NativeNetwork_mediaClose(JNIEnv *env, jobject thiz, jlong handle,
                                                        jlong stream_id) {
    (void)env;
    (void)thiz;
    return (jint)nanobot_network_media_close((int64_t)handle, (uint64_t)stream_id);
}

JNIEXPORT jint JNICALL
Java_ai_nanobot_mobile_network_NativeNetwork_mediaPurge(JNIEnv *env, jobject thiz, jlong handle) {
    (void)env;
    (void)thiz;
    return (jint)nanobot_network_media_purge((int64_t)handle);
}
