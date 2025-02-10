from openai import OpenAI
from mem0 import Memory
import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox, END
import markdown
import os

openai_client = OpenAI()

openai_client.api_key = os.getenv("OPENAI_API_KEY")
openai_client.base_url = os.getenv("OPENAI_BASE_URL")
config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o",
            "temperature": 0.2,
            "max_tokens": 1500,
            "openai_base_url": os.getenv("OPENAI_BASE_URL")
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "url": os.getenv("QDRANT_BASE_URL"),
            "port": 6333,
            "api_key": os.getenv("QDRANT_API_KEY"),
            "embedding_model_dims": 3072,
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-large",
            "embedding_dims": 3072,
            "openai_base_url": os.getenv("OPENAI_BASE_URL")
        }
    }
}
mem0 = Memory.from_config(config)

class ChatGUI:
    def __init__(self, master):
        self.master = master
        master.title("AI Chat with Memory")
        master.geometry("800x600")

        # 初始化 messages
        self.messages = [
            {"role": "system", "content": "You are a helpful AI. Answer based on query and memories."}
        ]

        # 创建聊天记录显示区域
        self.chat_history = scrolledtext.ScrolledText(
            master, 
            wrap=tk.WORD,
            state='disabled',
            font=('Arial', 12)
        )
        self.chat_history.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # 创建输入区域
        input_frame = ttk.Frame(master)
        input_frame.pack(padx=10, pady=5, fill=tk.X)

        self.user_input = ttk.Entry(
            input_frame, 
            font=('Arial', 12)
        )
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.user_input.bind("<Return>", self.send_message)

        send_btn = ttk.Button(
            input_frame,
            text="Send",
            command=self.send_message
        )
        send_btn.pack(side=tk.RIGHT, padx=(5,0))

        # 初始化对话历史
        self.update_chat_display("AI: Hello! How can I help you today?\n", "ai")
    
    def send_message(self, event=None):
        user_text = self.user_input.get().strip()
        if not user_text:
            return

        if user_text.lower() == 'exit':
            self.master.destroy()
            return
        self.update_chat_display(f"You: {user_text}\n", "user")
        self.user_input.delete(0, END)

        try:
            self.messages = chat_with_memories(user_text, gui=self, messages=self.messages)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to get response: {str(e)}")
        finally:
            self.update_chat_display(message="\n", sender="ai")
            self.master.update()

    def update_chat_display(self, message, sender):
        self.chat_history.configure(state='normal')
        self.chat_history.insert(END, message)
        self.chat_history.configure(state='disabled')
        self.chat_history.see(END)

def chat_with_memories(message: str, user_id: str = "default_user", gui: ChatGUI = None, messages=None) -> list:
    if messages is None:
        messages = [
            {"role": "system", "content": "You are a helpful AI. Answer based on query and memories."}
        ]

    # 保持原有记忆功能不变
    relevant_memories = mem0.search(query=message, user_id=user_id, limit=3)
    memories_str = "\n".join(f"- {entry['memory']}" for entry in relevant_memories)
    
    system_memories = f"User Memories:\n{memories_str}"
    messages.append({"role": "system", "content": system_memories})
    messages.append({"role": "user", "content": message})
    assistant_response = ""
    stream = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True  # 启用流式传输
    )
    for chunk in stream:
        if len(chunk.choices) > 0:
            delta = (chunk.choices[0].delta.content or "")
            assistant_response += delta
            # 实时更新聊天显示
            gui.update_chat_display(delta, "ai")
            gui.master.update()
    messages.append({"role": "assistant", "content": assistant_response})
    mem0.add(messages, user_id=user_id)
    return messages

if __name__ == "__main__":
    root = tk.Tk()
    gui = ChatGUI(root)
    root.mainloop()