
import asyncio
import functools
import inspect
import logging
import time

logger = logging.getLogger("ai_proj")


SENSITIVE_NAMES = (
    "password",
    "secret",
    "token",
    "api_key",
    "authorization",
    "credential",
)


def _mask_value(name, value):
    if any(
        sensitive in name.lower()
        for sensitive in SENSITIVE_NAMES
    ):
        return "********"

    return value


def _get_safe_arguments(func, args, kwargs):
    signature = inspect.signature(func)
    bound = signature.bind_partial(*args, **kwargs)

    safe_arguments = {}

    for name, value in bound.arguments.items():
        safe_arguments[name] = _mask_value(name, value)

    return safe_arguments


def log_function(
    func=None,
    *,
    log_args=False,
    log_result=False,
    slow_threshold=1.0,
):
    def decorator(target_func):

        function_name = target_func.__qualname__

        if asyncio.iscoroutinefunction(target_func):

            @functools.wraps(target_func)
            async def async_wrapper(*args, **kwargs):

                start_time = time.perf_counter()

                logger.debug(
                    "ENTRYPOINT | %s | Started",
                    function_name,
                )

                if log_args:
                    safe_args = _get_safe_arguments(
                        target_func,
                        args,
                        kwargs,
                    )
                    logger.debug(
                        "%s arguments: %s",
                        function_name,
                        safe_args,
                    )

                try:
                    result = await target_func(*args, **kwargs)

                    duration = time.perf_counter() - start_time

                    logger.debug(
                        "ENTRYPOINT | %s | Completed in %.3f seconds",
                        function_name,
                        duration,
                    )

                    if log_result:
                        logger.debug(
                            "%s result: %s",
                            function_name,
                            result,
                        )

                    if duration >= slow_threshold:
                        logger.warning(
                            "%s took %.3f seconds",
                            function_name,
                            duration,
                        )

                    return result

                except Exception:
                    duration = time.perf_counter() - start_time

                    logger.exception(
                        "ENTRYPOINT | %s | Failed after %.3f seconds",
                        function_name,
                        duration,
                    )
                    raise

            return async_wrapper

        @functools.wraps(target_func)
        def sync_wrapper(*args, **kwargs):

            start_time = time.perf_counter()

            logger.debug(
                "ENTRYPOINT | %s | Started",
                function_name,
            )

            if log_args:
                safe_args = _get_safe_arguments(
                    target_func,
                    args,
                    kwargs,
                )
                logger.debug(
                    "%s arguments: %s",
                    function_name,
                    safe_args,
                )

            try:
                result = target_func(*args, **kwargs)

                duration = time.perf_counter() - start_time

                logger.debug(
                    "ENTRYPOINT | %s | Completed in %.3f seconds",
                    function_name,
                    duration,
                )

                if log_result:
                    logger.debug(
                        "%s result: %s",
                        function_name,
                        result,
                    )

                if duration >= slow_threshold:
                    logger.warning(
                        "%s took %.3f seconds",
                        function_name,
                        duration,
                    )

                return result

            except Exception:
                duration = time.perf_counter() - start_time

                logger.exception(
                    "ENTRYPOINT | %s | Failed after %.3f seconds",
                    function_name,
                    duration,
                )
                raise

        return sync_wrapper

    if func is not None:
        return decorator(func)

    return decorator
