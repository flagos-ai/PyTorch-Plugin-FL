#include <include/flagos.h>
#include <acl/acl_rt.h>

Error_t StreamCreateWithPriority(Stream_t* stream, unsigned int flags, int priority) {
  aclError err = aclrtCreateStream((aclrtStream*)stream);
  return (err == ACL_SUCCESS) ? Success : ErrorUnknown;
}

Error_t StreamCreate(Stream_t* stream) {
  aclError err = aclrtCreateStream((aclrtStream*)stream);
  return (err == ACL_SUCCESS) ? Success : ErrorUnknown;
}

Error_t StreamGetPriority(Stream_t stream, int* priority) {
  *priority = 0;
  return Success;
}

Error_t StreamDestroy(Stream_t stream) {
  aclError err = aclrtDestroyStream((aclrtStream)stream);
  return (err == ACL_SUCCESS) ? Success : ErrorUnknown;
}

Error_t StreamQuery(Stream_t stream) {
  aclrtStreamStatus status;
  aclError err = aclrtStreamQuery((aclrtStream)stream, &status);
  if (err != ACL_SUCCESS) {
    return ErrorUnknown;
  }
  return (status == ACL_STREAM_STATUS_COMPLETE) ? Success : ErrorNotReady;
}

Error_t StreamSynchronize(Stream_t stream) {
  aclError err = aclrtSynchronizeStream((aclrtStream)stream);
  return (err == ACL_SUCCESS) ? Success : ErrorUnknown;
}

Error_t StreamWaitEvent(Stream_t stream, Event_t event, unsigned int flags) {
  aclError err = aclrtStreamWaitEvent((aclrtStream)stream, (aclrtEvent)event);
  return (err == ACL_SUCCESS) ? Success : ErrorUnknown;
}

Error_t EventCreateWithFlags(Event_t* event, unsigned int flags) {
  aclError err = aclrtCreateEvent((aclrtEvent*)event);
  return (err == ACL_SUCCESS) ? Success : ErrorUnknown;
}

Error_t EventCreate(Event_t* event) {
  aclError err = aclrtCreateEvent((aclrtEvent*)event);
  return (err == ACL_SUCCESS) ? Success : ErrorUnknown;
}

Error_t EventDestroy(Event_t event) {
  aclError err = aclrtDestroyEvent((aclrtEvent)event);
  return (err == ACL_SUCCESS) ? Success : ErrorUnknown;
}

Error_t EventRecord(Event_t event, Stream_t stream) {
  aclError err = aclrtRecordEvent((aclrtEvent)event, (aclrtStream)stream);
  return (err == ACL_SUCCESS) ? Success : ErrorUnknown;
}

Error_t EventSynchronize(Event_t event) {
  aclError err = aclrtSynchronizeEvent((aclrtEvent)event);
  return (err == ACL_SUCCESS) ? Success : ErrorUnknown;
}

Error_t EventQuery(Event_t event) {
  aclrtEventRecordedStatus status;
  aclError err = aclrtQueryEventStatus((aclrtEvent)event, &status);
  if (err != ACL_SUCCESS) {
    return ErrorUnknown;
  }
  return (status == ACL_EVENT_RECORDED_STATUS_COMPLETE) ? Success : ErrorNotReady;
}

Error_t EventElapsedTime(float* ms, Event_t start, Event_t end) {
  *ms = 0.0f;
  return Success;
}
