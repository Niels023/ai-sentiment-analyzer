import tkinter as tk
from tkinter import messagebox
from transformers import pipeline

class SentimentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI sentiment analyzer")
        self.root.geometry("400x300")

        print("Loading AI model... please wait...")
        self.analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        print("Model loaded!")

        self.label_title = tk.Label(root, text="Enter a sentence:", font=("Arial", 12, "bold"))
        self.label_title.pack(pady=10)

        self.text_entry = tk.Entry(root, width=40, font=("Arial", 11))
        self.text_entry.pack(pady=5)
        self.text_entry.bind('<Return>', self.analyze_text)

        self.btn_analyze = tk.Button(root, text="Analyze sentiment", command=self.analyze_text, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        self.btn_analyze.pack(pady=20)

        self.label_result = tk.Label(root, text="", font=("Helvetica", 14))
        self.label_result.pack(pady=10)

    def analyze_text(self, event=None):
        user_text = self.text_entry.get()

        if not user_text.strip():
            messagebox.showwarning("Warning", "Please type something first.")
            return

        try:
            prediction = self.analyzer(user_text)[0]
            label = prediction['label']
            score = prediction['score']

            result_text = f"sentiment: {label}\nConfidence: {score:.1%}"
            
            color = "green" if label == "POSITIVE" else "red"
            self.label_result.config(text=result_text, fg=color)
            
        except Exception as e:
            messagebox.showerror("Error", f"Something went wrong: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    
    app = SentimentApp(root)
    
    root.mainloop()
