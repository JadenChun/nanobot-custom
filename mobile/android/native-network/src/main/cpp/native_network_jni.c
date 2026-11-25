#include <jni.h>
#include <stdint.h>
#include <stdlib.h>

#include "nanobot_network.h"

/* The broker limits raw WebSocket frames to 40 MiB. Allow Base64 and JSON overhead. */
#define MAX_POLL_EVENT_BYTES (64 * 1024 * 1024)

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

static jint JNICALL jni_abiVersion(JNIEnv *env, jobject thiz) {
    (void)env;
    (void)thiz;
    return (jint)nanobot_network_abi_version();
}

static jlong JNICALL jni_start(JNIEnv *env, jobject thiz, jstring config) {
    (void)thiz;
    const char *raw = (*env)->GetStringUTFChars(env, config, NULL);
    if (raw == NULL) {
        return 0;
    }
    int64_t handle = nanobot_network_start(raw);
    (*env)->ReleaseStringUTFChars(env, config, raw);
    return (jlong)handle;
}

static jbyteArray JNICALL jni_status(JNIEnv *env, jobject thiz, jlong handle) {
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

static jbyteArray JNICALL jni_loginUrl(JNIEnv *env, jobject thiz, jlong handle) {
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

static jlong JNICALL jni_http(JNIEnv *env, jobject thiz, jlong handle,
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

static jint JNICALL jni_cancel(JNIEnv *env, jobject thiz, jlong handle,
                               jlong request_id) {
    (void)env;
    (void)thiz;
    return (jint)nanobot_network_cancel((int64_t)handle, (uint64_t)request_id);
}

static jlong JNICALL jni_wsOpen(JNIEnv *env, jobject thiz, jlong handle,
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

static jint JNICALL jni_wsSend(JNIEnv *env, jobject thiz, jlong handle,
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

static jint JNICALL jni_wsClose(JNIEnv *env, jobject thiz, jlong handle,
                                jlong ws_id) {
    (void)env;
    (void)thiz;
    return (jint)nanobot_network_ws_close((int64_t)handle, (uint64_t)ws_id);
}

static jint JNICALL jni_wsPing(JNIEnv *env, jobject thiz, jlong handle,
                               jlong ws_id) {
    (void)env;
    (void)thiz;
    return (jint)nanobot_network_ws_ping((int64_t)handle, (uint64_t)ws_id);
}

static jbyteArray JNICALL jni_pollEvent(JNIEnv *env, jobject thiz, jlong handle) {
    (void)thiz;
    int32_t capacity = 512 * 1024;
    char *buffer = malloc(capacity);
    if (buffer == NULL) {
        return NULL;
    }
    int32_t length = nanobot_network_poll_event((int64_t)handle, buffer, capacity);
    if (length == -7) {
        free(buffer);
        capacity = nanobot_network_pending_event_size((int64_t)handle);
        if (capacity <= 0 || capacity > MAX_POLL_EVENT_BYTES) {
            return NULL;
        }
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

static jint JNICALL jni_stop(JNIEnv *env, jobject thiz, jlong handle) {
    (void)env;
    (void)thiz;
    return (jint)nanobot_network_stop((int64_t)handle);
}

static jstring JNICALL jni_lastError(JNIEnv *env, jobject thiz) {
    (void)thiz;
    /* Copy under the library lock; never keep a library-owned pointer. */
    char buffer[2048];
    int32_t length = nanobot_network_last_error_copy(buffer, (int32_t)sizeof(buffer));
    if (length < 0) {
        length = 0;
    }
    buffer[length] = '\0';
    return (*env)->NewStringUTF(env, buffer);
}

static jint JNICALL jni_mediaSetToken(JNIEnv *env, jobject thiz,
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

static jlong JNICALL jni_mediaOpen(JNIEnv *env, jobject thiz, jlong handle,
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

static jbyteArray JNICALL jni_mediaInfo(JNIEnv *env, jobject thiz, jlong handle,
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

static jint JNICALL jni_mediaRead(JNIEnv *env, jobject thiz, jlong handle,
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

static jint JNICALL jni_mediaClose(JNIEnv *env, jobject thiz, jlong handle,
                                   jlong stream_id) {
    (void)env;
    (void)thiz;
    return (jint)nanobot_network_media_close((int64_t)handle, (uint64_t)stream_id);
}

static jint JNICALL jni_mediaPurge(JNIEnv *env, jobject thiz, jlong handle) {
    (void)env;
    (void)thiz;
    return (jint)nanobot_network_media_purge((int64_t)handle);
}

static jint JNICALL jni_networkChanged(JNIEnv *env, jobject thiz, jlong handle) {
    (void)env;
    (void)thiz;
    return (jint)nanobot_network_network_changed((int64_t)handle);
}

static const JNINativeMethod methods[] = {
    {"abiVersion", "()I", (void *)jni_abiVersion},
    {"start", "(Ljava/lang/String;)J", (void *)jni_start},
    {"status", "(J)[B", (void *)jni_status},
    {"loginUrl", "(J)[B", (void *)jni_loginUrl},
    {"http", "(JLjava/lang/String;)J", (void *)jni_http},
    {"cancel", "(JJ)I", (void *)jni_cancel},
    {"wsOpen", "(JLjava/lang/String;)J", (void *)jni_wsOpen},
    {"wsSend", "(JJI[B)I", (void *)jni_wsSend},
    {"wsClose", "(JJ)I", (void *)jni_wsClose},
    {"wsPing", "(JJ)I", (void *)jni_wsPing},
    {"pollEvent", "(J)[B", (void *)jni_pollEvent},
    {"mediaSetToken", "(JLjava/lang/String;)I", (void *)jni_mediaSetToken},
    {"mediaOpen", "(JLjava/lang/String;Ljava/lang/String;)J", (void *)jni_mediaOpen},
    {"mediaInfo", "(JJ)[B", (void *)jni_mediaInfo},
    {"mediaRead", "(JJ[B)I", (void *)jni_mediaRead},
    {"mediaClose", "(JJ)I", (void *)jni_mediaClose},
    {"mediaPurge", "(J)I", (void *)jni_mediaPurge},
    {"networkChanged", "(J)I", (void *)jni_networkChanged},
    {"stop", "(J)I", (void *)jni_stop},
    {"lastError", "()Ljava/lang/String;", (void *)jni_lastError},
};

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
    (void)reserved;
    JNIEnv *env;
    if ((*vm)->GetEnv(vm, (void **)&env, JNI_VERSION_1_6) != JNI_OK) {
        return JNI_ERR;
    }
    jclass type = (*env)->FindClass(env, JNI_CLASS_NAME);
    if (type == NULL) {
        return JNI_ERR;
    }
    jint result = (*env)->RegisterNatives(env, type, methods, sizeof(methods) / sizeof(methods[0]));
    (*env)->DeleteLocalRef(env, type);
    return result == JNI_OK ? JNI_VERSION_1_6 : JNI_ERR;
}
