from pd_ai_agent_core.core_types.llm_chat_ai_agent import (
    LlmChatAgent,
    LlmChatResult,
    LlmChatAgentResponse,
    AgentFunctionDescriptor,
)
from pd_ai_agent_core.services.service_registry import ServiceRegistry
from pd_ai_agent_core.services.notification_service import NotificationService
from pd_ai_agent_core.services.log_service import LogService

from pd_ai_agent_core.services.vm_datasource_service import VmDatasourceService
from pd_ai_agent_core.messages import (
    create_agent_function_call_chat_message,
    create_clean_agent_function_call_chat_message,
)
import json
import logging
from pd_ai_agent_core.parallels_desktop.execute_on_vm import execute_on_vm
from pd_ai_agent_core.helpers import (
    get_context_variable,
)
from pd_ai_agent_core.common import (
    NOTIFICATION_SERVICE_NAME,
    LOGGER_SERVICE_NAME,
    VM_DATASOURCE_SERVICE_NAME,
)
import openai
import requests
from typing import List
from pd_ai_developer_agent.llm_agents.helpers import get_vm_details

logger = logging.getLogger(__name__)


def ANALYSE_WEB_PAGE_LLM_PROMPT(html_content) -> str:
    result = """You are an assistant that analyses webpages for what the user is trying to achieve.
You will need to analyse the webpage and provide a summary of the content and the user's intent.
For example if the user is trying to create a new project, you will need to analyse the webpage and provide a summary of the content and the user's intent.
"""

    if html_content is not None:
        result += f"""Use the provided html content: {html_content}\

"""
    return result


def DEVELOPER_AGENT_PROMPT(context_variables) -> str:
    result = """You are an assistant that analyses webpages for programming content.
You need to extract the code snippets and the LLM analysis of the webpage.

You need to return a json object with the following keys so they can be used to create a VM
by the create_vm_agent, once done pass the json object to the create_vm_agent:
- os
- languages
- dependencies
- llm_summary
- a code structure with files and files content from what you have analysed

When generating the code structure, make sure to include all the files and files content from what you have analysed.
In some cases you might need to join multiple code blocks together to form a complete file.
Try to create a project that will work on the users request.

make sure if there is code in the webpage, you pass it to the create vm agent
so the machine can be created, you will fail your task if you don't pass the code to the create vm agent


"""
    if context_variables is not None:
        result += f"""Use the provided context in JSON format: {json.dumps(context_variables)}\
If the user has provided a vm id, use it to perform the operation on the VM.
If the user has provided a vm name, use it on your responses to the user to identify the VM instead of the vm id.

"""
    return result


DEVELOPER_AGENT_TRANSFER_INSTRUCTIONS = """
Call this function if the user is asking you to analyse a webpage or code in a webpage.
"""


class DeveloperAgent(LlmChatAgent):
    def __init__(self):
        super().__init__(
            name="Developer Agent",
            instructions=DEVELOPER_AGENT_PROMPT,
            description="This agent is responsible for analysing webpages for programming content.",
            functions=[self.analyse_updates_tool],  # type: ignore
            function_descriptions=[
                AgentFunctionDescriptor(
                    name=self.analyse_updates_tool.__name__,
                    description="Analyse the security of a VM",
                ),
            ],
            transfer_instructions=DEVELOPER_AGENT_TRANSFER_INSTRUCTIONS,
        )

    def fetch_webpage(self, url: str):
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Error fetching the webpage: {e}")
            return None

    def extract_code_snippets(html_content):
        soup = BeautifulSoup(html_content, "html.parser")
        code_snippets = []

        # Extract content from <code> and <pre> tags
        for tag in soup.find_all(["code", "pre"]):
            code_snippets.append(tag.get_text())

        return code_snippets

    def analyse_page_with_llm(html_content):
        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant that analyses webpages for programming content.",
                    },
                    {"role": "user", "content": html_content},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error using OpenAI API: {e}")
            return None

    def detect_language(code):
        try:
            lexer = guess_lexer(code)
            return lexer.name
        except ClassNotFound:
            return "Unknown"

    def infer_dependencies(code, language):
        dependencies = []

        if language == "Python":
            for line in code.splitlines():
                if line.startswith("import") or line.startswith("from"):
                    dependencies.append(line)
        elif language == "JavaScript":
            for line in code.splitlines():
                if line.startswith("require(") or "import" in line:
                    dependencies.append(line)
        # Add more languages and patterns as needed

        return dependencies

    def summarize_requirements(html_content, code_snippets, llm_analysis):
        summary = {
            "os": "Ubuntu",
            "languages": [],
            "dependencies": [],
            "llm_summary": llm_analysis,
        }

        for code in code_snippets:
            language = detect_language(code)
            if language != "Unknown" and language not in summary["languages"]:
                summary["languages"].append(language)

            dependencies = infer_dependencies(code, language)
            summary["dependencies"].extend(
                dep for dep in dependencies if dep not in summary["dependencies"]
            )

        return summary

    def agent_analyse_webpage_tool(
        session_context: dict, context_variables: dict, url: str
    ) -> LlmChatAgentResponse:
        """Analyse webpage content."""
        ns = ServiceRegistry.get(
            session_context["session_id"],
            NOTIFICATION_SERVICE_NAME,
            NotificationService,
        )
        ls = ServiceRegistry.get(
            session_context["session_id"], LOGGER_SERVICE_NAME, LogService
        )
        ls.info(
            session_context["channel"],
            f"Analyzing webpage with args {session_context}, {context_variables}, {url}",
        )
        ns.send_sync(
            create_agent_function_call_chat_message(
                session_id=session_context["session_id"],
                channel=session_context["channel"],
                name=f"Analyzing webpage ",
                arguments={},
                linked_message_id=session_context["linked_message_id"],
                is_partial=session_context["is_partial"],
            )
        )
        if not url:
            context_url = get_context_variable(
                "url", session_context, context_variables
            )
            if not context_url:
                raise RuntimeError("No URL provided")
            url = context_url

        ns.send_sync(
            create_agent_function_call_chat_message(
                session_id=session_context["session_id"],
                channel=session_context["channel"],
                name=f"Analyzing webpage {url}",
                arguments={},
                linked_message_id=session_context["linked_message_id"],
                is_partial=session_context["is_partial"],
            )
        )

        try:
            html_content = fetch_webpage(url)
            if not html_content:
                return LlmChatAgentResponse(
                    status="error",
                    message="Failed to fetch webpage",
                    error="Failed to fetch webpage",
                )

            code_snippets = extract_code_snippets(html_content)
            llm_analysis = analyse_page_with_llm(html_content)
            ns.send_sync(
                create_agent_function_call_chat_message(
                    session_id=session_context["session_id"],
                    channel=session_context["channel"],
                    name=f"Summarizing findings {url}",
                    arguments={},
                    linked_message_id=session_context["linked_message_id"],
                    is_partial=session_context["is_partial"],
                )
            )
            summary = summarize_requirements(html_content, code_snippets, llm_analysis)
            ns.send_sync(
                create_clean_agent_function_call_chat_message(
                    session_id=session_context["session_id"],
                    channel=session_context["channel"],
                    linked_message_id=session_context["linked_message_id"],
                    is_partial=session_context["is_partial"],
                )
            )
            return LlmChatAgentResponse(
                status="success",
                message=f"Analysed webpage: {url}",
                data={"summary": summary},
            )
        except Exception as e:
            ns.send_sync(
                create_clean_agent_function_call_chat_message(
                    session_id=session_context["session_id"],
                    channel=session_context["channel"],
                    linked_message_id=session_context["linked_message_id"],
                    is_partial=session_context["is_partial"],
                )
            )
            ls.exception(
                session_context["channel"],
                f"Failed to analyse webpage: {url}",
                e,
            )
            return LlmChatAgentResponse(
                status="error",
                message=f"Failed to analyse webpage: {url}",
                error=str(e),
            )
