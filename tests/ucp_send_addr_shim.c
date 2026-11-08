/* LD_PRELOAD shim for tests/test_host_coll_addresses.py: counts the distinct
 * host buffer addresses commux passes to ucp_tag_send_nbx, then forwards the
 * call. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdlib.h>
#include <ucp/api/ucp.h>

#define CAP 65536
static uintptr_t seen[CAP];
static long nseen, nsends;

long shim_distinct_host_send_addrs(void) { return nseen; }
long shim_host_sends(void) { return nsends; }
void shim_reset(void) {
  nseen = 0;
  nsends = 0;
}

ucs_status_ptr_t ucp_tag_send_nbx(ucp_ep_h ep, const void *buffer, size_t count,
                                  ucp_tag_t tag,
                                  const ucp_request_param_t *param) {
  static ucs_status_ptr_t (*real)(ucp_ep_h, const void *, size_t, ucp_tag_t,
                                  const ucp_request_param_t *);
  if (!real) { /* libucp is loaded RTLD_LOCAL by the extension: look it up by
                  soname */
    void *h = dlopen("libucp.so.0", RTLD_NOW | RTLD_NOLOAD);
    real =
        h ? dlsym(h, "ucp_tag_send_nbx") : dlsym(RTLD_NEXT, "ucp_tag_send_nbx");
    if (!real) abort();
  }
  int host = !(param && (param->op_attr_mask & UCP_OP_ATTR_FIELD_MEMORY_TYPE) &&
               param->memory_type != UCS_MEMORY_TYPE_HOST);
  if (host && count > 0) {
    long i;
    ++nsends;
    for (i = 0; i < nseen && seen[i] != (uintptr_t)buffer; ++i) {
    }
    if (i == nseen && nseen < CAP) seen[nseen++] = (uintptr_t)buffer;
  }
  return real(ep, buffer, count, tag, param);
}
