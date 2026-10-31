#include <jni.h>
#include <stdio.h>
#include <sys/utsname.h>
#include <unistd.h>

JNIEXPORT jstring JNICALL
Java_ai_nanobot_mobile_NativeProbe_nativeIdentity(JNIEnv *env, jobject thiz) {
    struct utsname info;
    if (uname(&info) != 0) {
        return (*env)->NewStringUTF(env, "unknown:-1");
    }

    long page_size = sysconf(_SC_PAGESIZE);
    char buffer[256];
    snprintf(buffer, sizeof(buffer), "%s:%ld", info.machine, page_size);
    return (*env)->NewStringUTF(env, buffer);
}
