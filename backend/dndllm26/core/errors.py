from __future__ import annotations


class AppError(Exception):
    code = "application_error"
    status_code = 500
    retryable = False

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class NotFoundError(AppError):
    code = "not_found"
    status_code = 404


class ConflictError(AppError):
    code = "conflict"
    status_code = 409


class ValidationError(AppError):
    code = "validation_error"
    status_code = 422


class OperationInProgressError(ConflictError):
    code = "operation_in_progress"
    retryable = True


class ModelUnavailableError(AppError):
    code = "model_unavailable"
    status_code = 503
    retryable = True


class ModelResponseError(AppError):
    code = "invalid_model_response"
    status_code = 502
    retryable = True


class LoreProcessingError(AppError):
    code = "lore_processing_error"
    status_code = 422
