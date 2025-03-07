from pd_ai_agent_core.services.service_registry import ServiceRegistry
from pd_ai_agent_core.services.notification_service import NotificationService
from pd_ai_agent_core.services.vm_datasource_service import VmDatasourceService
from pd_ai_agent_core.common import (
    NOTIFICATION_SERVICE_NAME,
    VM_DATASOURCE_SERVICE_NAME,
)
from pd_ai_agent_core.messages import (
    create_clean_agent_function_call_chat_message,
)
from pd_ai_agent_core.core_types.llm_chat_ai_agent import (
    LlmChatAgentResponse,
)
from pd_ai_agent_core.parallels_desktop.models.virtual_machine import VirtualMachine


def get_vm_details(
    session_context: dict, context_variables: dict, vm_id: str
) -> tuple[VirtualMachine | None, LlmChatAgentResponse | None]:
    ns = ServiceRegistry.get(
        session_context["session_id"],
        NOTIFICATION_SERVICE_NAME,
        NotificationService,
    )
    data = ServiceRegistry.get(
        session_context["session_id"],
        VM_DATASOURCE_SERVICE_NAME,
        VmDatasourceService,
    )
    if not data:
        ns.send_sync(
            create_clean_agent_function_call_chat_message(
                session_context["session_id"],
                session_context["channel"],
                session_context["linked_message_id"],
                session_context["is_partial"],
            )
        )
        return None, LlmChatAgentResponse(
            status="error",
            message="No vm datasource provided",
        )
    vm_details = data.datasource.get_vm(vm_id)
    if not vm_details:
        ns.send_sync(
            create_clean_agent_function_call_chat_message(
                session_context["session_id"],
                session_context["channel"],
                session_context["linked_message_id"],
                session_context["is_partial"],
            )
        )
        return None, LlmChatAgentResponse(
            status="error",
            message="No vm details provided",
        )
    return vm_details, None
