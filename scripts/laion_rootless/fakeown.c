/* LD_PRELOAD shim for rootless image builds inside a single-id user namespace (Leonardo has no /etc/subuid).
 * Only uid/gid 0 are mapped, so chown to any other id fails with EINVAL and setuid/setgroups to others fail.
 * Package managers (dpkg, apt) need those calls to succeed, not to take effect. Files stay owned by the build
 * user, which is root inside the image once it is run as an ordinary user; ownership info is not preserved.
 * Build: gcc -shared -fPIC -O2 -o fakeown.so fakeown.c   (glibc images only; not for musl/alpine) */
#define _GNU_SOURCE
#include <sys/types.h>
#include <unistd.h>
#include <grp.h>
int chown(const char *p, uid_t u, gid_t g) { return 0; }
int lchown(const char *p, uid_t u, gid_t g) { return 0; }
int fchown(int fd, uid_t u, gid_t g) { return 0; }
int fchownat(int d, const char *p, uid_t u, gid_t g, int f) { return 0; }
int setuid(uid_t u) { return 0; }
int setgid(gid_t g) { return 0; }
int seteuid(uid_t u) { return 0; }
int setegid(gid_t g) { return 0; }
int setreuid(uid_t a, uid_t b) { return 0; }
int setregid(gid_t a, gid_t b) { return 0; }
int setresuid(uid_t a, uid_t b, uid_t c) { return 0; }
int setresgid(gid_t a, gid_t b, gid_t c) { return 0; }
int setgroups(size_t n, const gid_t *l) { return 0; }
int initgroups(const char *u, gid_t g) { return 0; }
