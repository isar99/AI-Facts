import random
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from rich import print

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-70B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)


TOPICS = [
    "Black holes",
    "Ancient Egypt",
    "Deep sea creatures",
    "Time travel paradoxes",
    "Quantum physics",
    "Dinosaurs",
    "Mars exploration",
    "AI in everyday life",
    "Strange weather phenomena",
    "Myths from around the world",
    "Human brain mysteries",
    "Future transportation",
    "Volcanoes",
    "History of coffee",
    "Weird inventions",
    "Space telescopes",
    "Bioluminescence",
    "Lost civilizations",
    "Cryptography",
    "Extreme sports science"
]


class Fact(BaseModel):
    fact1: str = Field(description="Fact1 about a topic")
    fact2: str = Field(description="Fact2 about a topic")
    fact3: str = Field(description="Fact3 about a topic")
    fact4: str = Field(description="Fact4 about a topic")
    fact5: str = Field(description="Fact5 about a topic")
    fact6: str = Field(description="Fact6 about a topic")
    fact7: str = Field(description="Fact7 about a topic")
    fact8: str = Field(description="Fact8 about a topic")
    fact9: str = Field(description="Fact9 about a topic")
    fact10: str = Field(description="Fact10 about a topic")

parser = PydanticOutputParser(pydantic_object=Fact)

template = PromptTemplate(
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
    template="Generate 10 facts about {topic}. \n"
    "Return ONLY a valid JSON object.\n"
    "Do not add markdown.\n"
    "Do not add explanation text.\n"
    "Do not wrap the JSON in any extra object.\n"
    "Do not return a function call or schema.\n"
    "{format_instructions}"
)

chain = template | model | parser

topic = random.choice(TOPICS)

print(f"[bold green]Generating facts about:[/bold green] [bold purple]{topic}[/bold purple]\n")

result = chain.invoke({
    "topic": topic,
    })

print(result)
