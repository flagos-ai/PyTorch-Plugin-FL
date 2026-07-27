#include <include/flagos.h>
#include <tx_runtime.h>

Error_t StreamCreateWithPriority(
    Stream_t* stream,
    unsigned int /*flags*/,
    int /*priority*/) {
  // TX runtime has no stream priority support
  txError_t err = txStreamCreate(reinterpret_cast<txStream_t*>(stream));
  return (err == TX_SUCCESS) ? Success : ErrorUnknown;
}

Error_t StreamCreate(Stream_t* stream) {
  txError_t err = txStreamCreate(reinterpret_cast<txStream_t*>(stream));
  return (err == TX_SUCCESS) ? Success : ErrorUnknown;
}

Error_t StreamGetPriority(Stream_t /*stream*/, int* priority) {
  // TX runtime has no stream priority query
  *priority = 0;
  return Success;
}

Error_t StreamDestroy(Stream_t stream) {
  txError_t err = txStreamDestroy(reinterpret_cast<txStream_t>(stream));
  return (err == TX_SUCCESS) ? Success : ErrorUnknown;
}

Error_t StreamQuery(Stream_t stream) {
  txError_t err = txStreamQuery(reinterpret_cast<txStream_t>(stream));
  if (err == TX_SUCCESS) {
    return Success;
  } else if (err == TX_ERROR_NOT_READY) {
    return ErrorNotReady;
  }
  return ErrorUnknown;
}

Error_t StreamSynchronize(Stream_t stream) {
  txError_t err = txStreamSynchronize(reinterpret_cast<txStream_t>(stream));
  return (err == TX_SUCCESS) ? Success : ErrorUnknown;
}

Error_t StreamWaitEvent(Stream_t stream, Event_t event, unsigned int /*flags*/) {
  // TX runtime's StreamWaitEvent has no flags parameter
  txError_t err = txStreamWaitEvent(
      reinterpret_cast<txStream_t>(stream),
      reinterpret_cast<txEvent_t>(event));
  return (err == TX_SUCCESS) ? Success : ErrorUnknown;
}

Error_t EventCreateWithFlags(Event_t* event, unsigned int /*flags*/) {
  // TX runtime has no event flags support
  txError_t err = txEventCreate(reinterpret_cast<txEvent_t*>(event));
  return (err == TX_SUCCESS) ? Success : ErrorUnknown;
}

Error_t EventCreate(Event_t* event) {
  txError_t err = txEventCreate(reinterpret_cast<txEvent_t*>(event));
  return (err == TX_SUCCESS) ? Success : ErrorUnknown;
}

Error_t EventDestroy(Event_t event) {
  txError_t err = txEventDestroy(reinterpret_cast<txEvent_t>(event));
  return (err == TX_SUCCESS) ? Success : ErrorUnknown;
}

Error_t EventRecord(Event_t event, Stream_t stream) {
  txError_t err = txEventRecord(
      reinterpret_cast<txEvent_t>(event),
      reinterpret_cast<txStream_t>(stream));
  return (err == TX_SUCCESS) ? Success : ErrorUnknown;
}

Error_t EventSynchronize(Event_t event) {
  txError_t err = txEventSynchronize(reinterpret_cast<txEvent_t>(event));
  return (err == TX_SUCCESS) ? Success : ErrorUnknown;
}

Error_t EventQuery(Event_t event) {
  txError_t err = txEventQuery(reinterpret_cast<txEvent_t>(event));
  if (err == TX_SUCCESS) {
    return Success;
  } else if (err == TX_ERROR_NOT_READY) {
    return ErrorNotReady;
  }
  return ErrorUnknown;
}

Error_t EventElapsedTime(float* ms, Event_t start, Event_t end) {
  txError_t err = txEventElapsedTime(
      ms,
      reinterpret_cast<txEvent_t>(start),
      reinterpret_cast<txEvent_t>(end));
  return (err == TX_SUCCESS) ? Success : ErrorUnknown;
}