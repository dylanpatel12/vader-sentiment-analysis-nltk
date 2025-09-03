# Sentiment Analyzer with GUI (Tkinter + NLTK VADER)
# Features: Single text analysis + CSV batch analysis

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download vader_lexicon if not already installed
nltk.download('vader_lexicon')

# Create sentiment analyzer
sia = SentimentIntensityAnalyzer()

# ---- Functions ----
def analyze_text():
    """Analyze single input text from textbox"""
    text = entry.get("1.0", tk.END).strip()
    if not text:
        result_label.config(text="⚠️ Please enter some text.")
        return
    
    score = sia.polarity_scores(text)['compound']
    if score > 0.05:
        sentiment = "Positive 😀"
    elif score < -0.05:
        sentiment = "Negative 😞"
    else:
        sentiment = "Neutral 😐"
    
    result_label.config(
        text=f"Sentiment: {sentiment}\nScore: {score:.3f}"
    )

def analyze_csv():
    """Load and analyze sentiments from a CSV file"""
    file_path = filedialog.askopenfilename(
        filetypes=[("CSV Files", "*.csv")],
        title="Select a CSV file"
    )
    if not file_path:
        return
    
    try:
        df = pd.read_csv(file_path)
        if "review_text" not in df.columns:
            messagebox.showerror("Error", "CSV must have a 'review_text' column.")
            return
        
        # Apply sentiment analysis
        def analyze_sentiment(text):
            score = sia.polarity_scores(str(text))['compound']
            if score > 0.05:
                sentiment = 'Positive'
            elif score < -0.05:
                sentiment = 'Negative'
            else:
                sentiment = 'Neutral'
            return sentiment, score
        
        df['sentiment'], df['compound_score'] = zip(*df['review_text'].apply(analyze_sentiment))
        
        # Show summary
        summary = df['sentiment'].value_counts().to_string()
        messagebox.showinfo("CSV Analysis Complete", f"Summary:\n\n{summary}")
        
        # Save results
        output_path = file_path.replace(".csv", "_with_sentiment.csv")
        df.to_csv(output_path, index=False)
        messagebox.showinfo("Saved", f"Results saved to:\n{output_path}")
    
    except Exception as e:
        messagebox.showerror("Error", f"Failed to process file:\n{e}")

# ---- GUI ----
root = tk.Tk()
root.title("Sentiment Analyzer (VADER)")
root.geometry("500x400")
root.resizable(False, False)

# Title
title_label = ttk.Label(root, text="Sentiment Analyzer", font=("Arial", 14, "bold"))
title_label.pack(pady=10)

# Text entry
entry_label = ttk.Label(root, text="Enter text to analyze:")
entry_label.pack()
entry = tk.Text(root, height=5, width=55, font=("Arial", 11))
entry.pack(pady=5)

# Buttons
btn_frame = ttk.Frame(root)
btn_frame.pack(pady=10)

analyze_button = ttk.Button(btn_frame, text="Analyze Text", command=analyze_text)
analyze_button.grid(row=0, column=0, padx=10)

csv_button = ttk.Button(btn_frame, text="Analyze CSV", command=analyze_csv)
csv_button.grid(row=0, column=1, padx=10)

# Result label
result_label = ttk.Label(root, text="", font=("Arial", 12, "bold"), foreground="blue")
result_label.pack(pady=15)

# Run GUI
root.mainloop()
