import json
import re
from openai import OpenAI

# ═══════════════════════════════════════════════
#  Конфигурация
# ═══════════════════════════════════════════════
MODEL = "deepseek-r1"
MAX_TOKENS = 50000

client = OpenAI(
    api_key="sk-aitunnel-6nSOCdFD2jUgDD3fzNwfJtqFbtQl8BaL",       # ← Вставь свой ключ
    base_url="https://api.aitunnel.ru/v1/",
)

# ═══════════════════════════════════════════════
#  Инструменты (Tools)
# ═══════════════════════════════════════════════
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Узнать текущую погоду в указанном городе",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Название города, например: Москва"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Выполнить математическое вычисление",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Математическое выражение, например: 2 + 2 * 3"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Поиск информации в интернете",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# ═══════════════════════════════════════════════
#  Реализация инструментов (заглушки)
# ═══════════════════════════════════════════════
def get_weather(city: str) -> str:
    """Заглушка — замени на реальное API (OpenWeather, wttr.in и т.д.)"""
    fake_data = {
        "Москва": "+22°C, солнечно",
        "Санкт-Петербург": "+17°C, пасмурно",
        "Казань": "+25°C, ясно",
    }
    return json.dumps({
        "city": city,
        "weather": fake_data.get(city, f"+20°C, переменная облачность (данные для {city})")
    }, ensure_ascii=False)


def calculator(expression: str) -> str:
    """Безопасный калькулятор"""
    try:
        # Разрешаем только цифры и базовые операторы
        if not re.match(r'^[\d\s\+\-\*\/\.\(\)]+$', expression):
            return json.dumps({"error": "Недопустимое выражение"})
        result = eval(expression)  # безопасно после regex-проверки
        return json.dumps({"expression": expression, "result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


def search_web(query: str) -> str:
    """Заглушка — замени на реальное API (DuckDuckGo, SerpAPI и т.д.)"""
    return json.dumps({
        "query": query,
        "result": f"Результаты поиска по запросу «{query}»: (замени заглушку на реальный API)"
    }, ensure_ascii=False)


# Маппинг имён функций на реальные вызовы
TOOL_HANDLERS = {
    "get_weather": get_weather,
    "calculator": calculator,
    "search_web": search_web,
}

# ═══════════════════════════════════════════════
#  Системный промпт
# ═══════════════════════════════════════════════
SYSTEM_PROMPT = """\
Ты — полезный ИИ-ассистент. Отвечай кратко и по делу.
Если тебе нужны данные (погода, вычисления, информация из интернета) — 
используй доступные инструменты, а не выдумывай ответы.
Отвечай на русском языке.
"""

# ═══════════════════════════════════════════════
#  Ядро агента
# ═══════════════════════════════════════════════
class Agent:
    def __init__(self):
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _call_llm(self):
        """Отправить запрос в LLM и получить ответ (с возможными tool_calls)."""
        return client.chat.completions.create(
            model=MODEL,
            messages=self.messages,
            tools=tools,
            max_tokens=MAX_TOKENS,
        )

    def _handle_tool_calls(self, tool_calls):
        """Выполнить каждый tool_call и добавить результат в историю."""
        for call in tool_calls:
            func_name = call.function.name
            args = json.loads(call.function.arguments)

            print(f"  🔧 Вызываю инструмент: {func_name}({args})")

            handler = TOOL_HANDLERS.get(func_name)
            if handler:
                result = handler(**args)
            else:
                result = json.dumps({"error": f"Неизвестный инструмент: {func_name}"})

            # Добавляем результат инструмента в контекст диалога
            self.messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            })

    def chat(self, user_message: str) -> str:
        """Главный цикл: отправляем сообщение → обрабатываем tool_calls → получаем ответ."""
        self.messages.append({"role": "user", "content": user_message})

        max_rounds = 5  # защита от бесконечных циклов tool_calls
        for _ in range(max_rounds):
            response = self._call_llm()
            choice = response.choices[0]
            message = choice.message

            # Добавляем сообщение ассистента в историю
            self.messages.append(message.model_dump())

            # Если LLM не запрашивает инструменты — это финальный ответ
            if not message.tool_calls:
                # Убираем think-блоки для DeepSeek-R1
                content = message.content or ""
                return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

            # Иначе выполняем инструменты и продолжаем цикл
            print("  🤔 Агент думает и использует инструменты...")
            self._handle_tool_calls(message.tool_calls)

        return "⚠️ Превышено количество итераций агента."

    def reset(self):
        """Очистить историю диалога."""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]


# ═══════════════════════════════════════════════
#  REPL (интерактивный режим)
# ═══════════════════════════════════════════════
def main():
    agent = Agent()

    print("=" * 50)
    print("  🤖 AI Агент запущен")
    print("  Напиши 'выход' или 'quit' чтобы выйти")
    print("  Напиши 'сброс' чтобы очистить историю")
    print("=" * 50)

    while True:
        user_input = input("\n👤 Ты: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("выход", "quit", "exit"):
            print("👋 До свидания!")
            break

        if user_input.lower() == "сброс":
            agent.reset()
            print("🔄 История очищена.")
            continue

        print("🤖 Агент: ", end="", flush=True)
        try:
            reply = agent.chat(user_input)
            print(reply)
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")


if __name__ == "__main__":
    main()