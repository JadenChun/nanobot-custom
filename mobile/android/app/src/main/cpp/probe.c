#include <jni.h>
#include <stdio.h>
#include <sys/utsname.h>
#include <unistd.h>

static jstring JNICALL jni_nativeIdentity(JNIEnv *env, jobject thiz) {
    (void)thiz;
    struct utsname info;
    if (uname(&info) != 0) {
        return (*env)->NewStringUTF(env, "unknown:-1");
    }

    long page_size = sysconf(_SC_PAGESIZE);
    char buffer[256];
    snprintf(buffer, sizeof(buffer), "%s:%ld", info.machine, page_size);
    return (*env)->NewStringUTF(env, buffer);
}

static const JNINativeMethod methods[] = {
    {"nativeIdentity", "()Ljava/lang/String;", (void *)jni_nativeIdentity},
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
