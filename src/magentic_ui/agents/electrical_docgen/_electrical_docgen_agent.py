from autogen_agentchat.agents import BaseChatAgent
import os

from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Sequence,
    Union,
    AsyncGenerator,
)
from autogen_agentchat.base import Response
from pydantic import BaseModel
from autogen_core import CancellationToken, Component, ComponentModel
from autogen_agentchat.messages import (
    ModelClientStreamingChunkEvent,
)
from autogen_core.model_context import (
    ChatCompletionContext,
    UnboundedChatCompletionContext,
)
from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    SystemMessage,
)
from autogen_agentchat.utils import remove_images
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.messages import (
    BaseAgentEvent,
    BaseChatMessage,
    TextMessage,
    HandoffMessage,
    StructuredMessageFactory,
    ThoughtEvent,
)

from docxtpl import DocxTemplate
from pathlib import Path
import json


class GenDocxUseTemplate(object):
    """use template to generate doxc"""

    def __init__(self, template_file_path: str, output_file_path: str):
        self.template_file_path = template_file_path
        self.template_docx = DocxTemplate(Path(self.template_file_path))  # TODO
        self.output_file_path = Path(output_file_path)

    def gen_docx(self, variable_dict: Dict[str, Any], otput_file_name: str):
        self.template_docx.render(variable_dict)
        self.template_docx.save(self.output_file_path / otput_file_name)


class ElectrialcalDocGenConfig(BaseModel):
    """The declarative configuration for the ElectrialcalDocGen agent."""

    # pydantic 提供了具体的数据验证和序列化功能

    name: str
    model_client: ComponentModel
    tools: List[ComponentModel] | None = None
    model_context: ComponentModel | None = None
    description: str
    system_message: str | None = None
    model_client_stream: bool = False
    structured_message_factory: ComponentModel | None = None


class ElectrialcalDocGenAgent(BaseChatAgent, Component[ElectrialcalDocGenConfig]):
    """Electrical Documentation Generation Agent

    Core capabilities:
    ------------------
    - Generate documentation for electrical systems
    - Accept plain-language electrical design requirements
    usage example:
        electrialcal_docgen_agent = ElectrialcalDocGenAgent()
    """

    component_config_schema = ElectrialcalDocGenConfig
    component_provider_override = "magentic_ui.users._electriacal_docgen_agent"

    def __init__(
        self,
        name: str,
        model_client: ChatCompletionClient,
        work_dir: Path | str = "/workspace",
        bind_dir: Path | str | None = None,
        *,
        description: str = "An agent that provides assistance with ability to use tools.",
        system_message: (
            str | None
        ) = "You are a helpful AI assistant. Solve tasks using your tools. Reply with TERMINATE when the task has been completed.",
        model_client_stream: bool = False,
        model_context: ChatCompletionContext | None = None,
        output_content_type: type[BaseModel] | None = None,
        output_content_type_format: str | None = None,
    ):
        """
        Initialize the electrialcalDocGen agent.

        Args:
            name (str): The name of the agent.
        """
        super().__init__(name=name, description=description)
        self.work_dir = work_dir
        self.bind_dir = bind_dir
        self.model_client = model_client
        self.model_client_stream = model_client_stream
        self._system_messages: List[SystemMessage] = []
        if system_message is None:
            self._system_messages = []
        else:
            self._system_messages = [SystemMessage(content=system_message)]

        if model_context is not None:
            self._model_context = model_context
        else:
            self._model_context = UnboundedChatCompletionContext()

        self._output_content_type: type[BaseModel] | None = output_content_type
        self._output_content_type_format = output_content_type_format
        if output_content_type is not None:
            self._structured_message_factory = StructuredMessageFactory(
                input_model=output_content_type,
                format_string=output_content_type_format,
            )

        # docx gentator correlation
        # TODO this is temp code
        current_file_path = __file__
        self.current_dir_os_path = os.path.dirname(os.path.abspath(current_file_path))
        # create a docx generator
        self.generator = GenDocxUseTemplate(
            os.path.join(
                self.current_dir_os_path, "docx_template/0_系统部件方案设计说明书.docx"
            ),
            str(self.work_dir),
        )
        self._variable_dict = {}

        # if output_content_type is not None:
        #     self._structured_message_factory = StructuredMessageFactory(
        #         input_model=output_content_type, format_string = output_content_type_format
        #     )

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    async def on_messages(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        async for message in self.on_messages_stream(messages, cancellation_token):
            if isinstance(message, Response):
                return message
        raise AssertionError("The stream should have returned the final result.")

    def _clean_response_content(self, content: str) -> str:
        content = content.strip()
        # Remove thinking markers if present
        if "</think>" in content:
            content = content.split("</think>")[-1]
            content = content.strip()
        # Remove markdown code block markers
        if "```json" in content:
            content = content.split("```json")[-1]
            content = content.strip()

        keywords = ["```", "markdown"]  # remove keywords of content in start and end
        for keyword in keywords:
            if content.startswith(keyword):
                content = content[len(keyword) :].strip()
            if content.endswith(keyword):
                content = content[: -len(keyword)].strip()
        return content.strip()

    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        """
        Process the incoming messages with the ElectrialcalDocGen agent and yield events/responses as they happen.
        """

        # TODO delete this
        await self._add_messages_to_context(
            model_context=self._model_context,
            messages=messages,
        )

        inner_messages: List[BaseAgentEvent | BaseChatMessage] = []
        # TODO add jugement is contain all requirement message
        # first step: jugement is contain all requirement message
        task_context = """
        请根据用户的问题，判断生成需求说明书所需要的信息是否完善。
        当前需要验证的必填信息包括：
        1. 需求说明书的名称（例如：功率板设计需求说明书）
        2. 作者的名字（例如：张三）

        请按照以下规则进行判断：
        - 如果信息完善，回复格式为："信息完善"
        - 如果信息不完善，回复第一句必须包含"信息不完善"，然后列出缺失的信息项，格式为："信息不完善，还需要提供[缺失信息1]、[缺失信息2]
        """
        model_result = None
        async for inference_output in self._call_llm(
            model_client=self.model_client,
            model_client_stream=self.model_client_stream,
            system_messages=self._system_messages
            + [SystemMessage(content=task_context)],
            model_context=self._model_context,
            agent_name=self.name,
            cancellation_token=cancellation_token,
            output_content_type=self._output_content_type,
        ):
            if isinstance(inference_output, CreateResult):
                model_result = inference_output
            else:
                # Streaming chunk event
                yield inference_output

        assert model_result is not None, "No model result was produced."
        print(
            f"=========model_result is {self._clean_response_content(str(model_result.content))}"
        )
        if self._validate_response_integrity(
            self._clean_response_content(str(model_result.content))
        ):
            yield Response(
                chat_message=TextMessage(
                    content=self._clean_response_content(str(model_result.content)),
                    source=self.name,
                    models_usage=model_result.usage,
                ),
                inner_messages=[],
            )
            print(
                "需求信息不完善，请补充信息::",
                self._clean_response_content(str(model_result.content)),
            )
            return

        # second step: generate docx requrment content
        model_result = None
        task_context = """
        # 请严格根据以下规则处理用户输入：

        1. 从用户问题中精确提取需求说明的核心名称
        2. 输出格式：仅包含提取出的名称文本，不添加任何后缀、前缀或其他内容
        3. 如果无法识别或确定名称，输出："无法确定"

        # 禁止行为：
        - 添加"需求说明书"、"文档"等类型后缀
        - 包含解释性文字、问候语或额外信息
        - 修改或美化提取出的原始名称

        # 示例：
        输入："我需要一个功率板设计的需求说明书"
        输出："功率板设计"

        输入："请创建用户管理系统需求说明"
        输出："用户管理系统"

        输入："随便聊聊"
        输出："无法确定"
        """
        async for inference_output in self._call_llm(
            model_client=self.model_client,
            model_client_stream=self.model_client_stream,
            system_messages=self._system_messages
            + [SystemMessage(content=task_context)],
            model_context=self._model_context,
            agent_name=self.name,
            cancellation_token=cancellation_token,
            output_content_type=self._output_content_type,
        ):
            if isinstance(inference_output, CreateResult):
                model_result = inference_output
            else:
                # Streaming chunk event
                yield inference_output
        assert model_result is not None, "No model result was produced."
        print(
            f"=========model_result is {str(model_result.content)},remove thinking is {self._clean_response_content(str(model_result.content))}"
        )

        self._variable_dict["_coverpage_Project_Name"] = self._clean_response_content(
            str(model_result.content)
        )

        if model_result.thought:
            thought_event = ThoughtEvent(content=model_result.thought, source=self.name)
            yield thought_event
            inner_messages.append(thought_event)

        # await self._model_context.add_message(
        #     AssistantMessage(
        #         content=model_result.content,
        #         source=self.name,
        #         thought=getattr(model_result, "thought", None),
        #         )
        #     )
        self.generator.gen_docx(self._variable_dict, "output.docx")

        yield Response(
            chat_message=TextMessage(
                content=f"electriacal docgen is complete!",
                source=self.name,
                models_usage=model_result.usage,
            ),
            inner_messages=[],
        )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset the assistant agent to its initialization state."""
        await self._model_context.clear()

    def _validate_response_integrity(self, response_text: str) -> bool:
        """
        validate response is contains 信息不完善
        return : bool
        TODO : current method is not perfect, need to be improved
        """
        result = False
        # check if response contains "信息完善"
        if "信息完善" in response_text and "不完善" not in response_text:
            return result

        # check if response contains "信息不完善"
        if "信息不完善" in response_text:
            result = True
        return result

    @staticmethod
    async def _add_messages_to_context(
        model_context: ChatCompletionContext,
        messages: Sequence[BaseChatMessage],
    ) -> None:
        """
        Add incoming messages to the model context.
        """
        for msg in messages:
            if isinstance(msg, HandoffMessage):
                for llm_msg in msg.context:
                    await model_context.add_message(llm_msg)
            await model_context.add_message(msg.to_model_message())

    @staticmethod
    def _get_compatible_context(
        model_client: ChatCompletionClient, messages: List[LLMMessage]
    ) -> Sequence[LLMMessage]:
        """Ensure that the messages are compatible with the underlying client, by removing images if needed."""
        if model_client.model_info["vision"]:
            return messages
        else:
            return remove_images(messages)

    @classmethod
    async def _call_llm(
        cls,
        model_client: ChatCompletionClient,
        model_client_stream: bool,
        system_messages: List[SystemMessage],
        model_context: ChatCompletionContext,
        agent_name: str,
        cancellation_token: CancellationToken,
        output_content_type: type[BaseModel] | None,
    ) -> AsyncGenerator[Union[CreateResult, ModelClientStreamingChunkEvent], None]:
        """
        Perform a model inference and yield either streaming chunk events or the final CreateResult.
        """

        all_messages = await model_context.get_messages()

        llm_messages = cls._get_compatible_context(
            model_client=model_client, messages=system_messages + all_messages
        )

        if model_client_stream:

            model_result: Optional[CreateResult] = None
            async for chunk in model_client.create_stream(
                llm_messages,
                tools=[],
                json_output=output_content_type,
                cancellation_token=cancellation_token,
            ):
                if isinstance(chunk, CreateResult):
                    model_result = chunk
                elif isinstance(chunk, str):
                    yield ModelClientStreamingChunkEvent(
                        content=chunk, source=agent_name
                    )
                else:
                    raise RuntimeError(f"Invalid chunk type: {type(chunk)}")
            if model_result is None:
                raise RuntimeError("No final model result in streaming mode.")
            yield model_result
        else:
            model_result = await model_client.create(
                llm_messages,
                tools=[],
                cancellation_token=cancellation_token,
                json_output=output_content_type,
            )
            yield model_result


async def main():

    from autogen_ext.models.openai import OpenAIChatCompletionClient
    from autogen_core.models import ModelFamily
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.conditions import TextMentionTermination

    model_client = OpenAIChatCompletionClient(
        model="qwq-32b",
        base_url="http://36.103.239.236:8000/v1/",
        api_key="placeholder",
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "family": ModelFamily.R1,
            "structured_output": True,
        },
    )
    electrial_gendoc = ElectrialcalDocGenAgent(
        "electrial_gendoc",
        model_client,
        work_dir="./docx_template",
        description="""你是一个docx文档生成agent,
负责生成项目所需要的文档，例如需求分析说明书。执行结束后会返回生成docx文档，代表着文档已经生成完毕。
切记，如果有相关于文档生成的需求，例如需求分析说明书，则通过此agent就可以独立完成任务，而不需要其他agent来完成。也不需要在web上进行联网搜索。""",
        system_message=(
            """你是一个docx文档生成agent,
负责生成项目所需要的文档，例如需求分析说明书。执行结束后会返回生成docx文档，代表着文档已经生成完毕。
切记，如果有相关于文档生成的需求，例如需求分析说明书，则通过此agent就可以独立完成任务，而不需要其他agent来完成。也不需要在web上进行联网搜索。"""
        ),
        model_client_stream=True,
    )

    # Create the critic agent.
    critic_agent = AssistantAgent(
        "critic",
        model_client=model_client,
        system_message="接收到完成生成docx的消息就代表任务完成了. Respond with 'APPROVE' to when your feedbacks are addressed.",
    )

    # async for msg in electrial_gendoc.on_messages_stream(
    #     [TextMessage(content="What is the weather in New York? only reply keypoint message", source="user")],
    #     cancellation_token=CancellationToken()
    # ):
    #     if isinstance(msg,Response):
    #         print(msg.chat_message.content)

    text_termination = TextMentionTermination("APPROVE")

    # Create a team with the primary and critic agents. primary_agent, critic_agent,
    team = RoundRobinGroupChat(
        [electrial_gendoc], termination_condition=text_termination
    )
    # Use `asyncio.run(...)` when running in a script.
    result = await team.run(
        task="帮我生成一个关于电机驱动电路的说明书.作者的名称为张三。"
    )
    print("final result: ", result)


if __name__ == "__main__":
    print("run _electrical_docgen_agent.py")
    import asyncio

    asyncio.run(main())
