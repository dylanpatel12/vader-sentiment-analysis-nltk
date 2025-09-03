from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QPushButton, QFileDialog, QMessageBox
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt
import sys
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER if missing
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()


class SentimentApp(QWidget):
    def __init__(self):
        super().__init__()

        # window 
        self.setWindowTitle("Sentiment Analyzer")
        self.resize(700, 500)

        # layout 
        layout = QVBoxLayout()
        self.setLayout(layout)

        # title
        title = QLabel("Sentiment Analyzer (VADER)")
        title.setFont(QFont("Arial Rounded MT Bold", 22))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # input box
        self.textbox = QTextEdit()
        self.textbox.setPlaceholderText("Type or paste text here...")
        self.textbox.setFixedHeight(120)
        layout.addWidget(self.textbox)

        # buttons
        btn_layout = QHBoxLayout()
        layout.addLayout(btn_layout)

        analyze_btn = QPushButton("🔍 Analyze Text")
        analyze_btn.clicked.connect(self.analyze_text)
        btn_layout.addWidget(analyze_btn)

        csv_btn = QPushButton("📂 Analyze CSV")
        csv_btn.clicked.connect(self.analyze_csv)
        btn_layout.addWidget(csv_btn)

        # resuts label
        self.result_label = QLabel("Result will appear here...")
        self.result_label.setFont(QFont("Arial", 14))
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.result_label)

        self.setStyleSheet("""
            QWidget {
                background-color: #121212;
                color: #eeeeee;
            }
            QLabel#title {
                color: #00ffc6;
                font-size: 26px;
                font-weight: bold;
            }
            QTextEdit {
                background: #1e1e1e;
                border: 2px solid #00ffc6;
                border-radius: 12px;
                padding: 10px;
                font-size: 14px;
            }
            QPushButton {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00c6ff, stop:1 #0072ff
                );
                color: white;
                border-radius: 12px;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0072ff, stop:1 #00c6ff
                );
            }
            QLabel {
                font-size: 14px;
            }
        """)

    def analyze_text(self):
        text = self.textbox.toPlainText().strip()
        if not text:
            self.result_label.setText(" Please enter some text.")
            return

        score = sia.polarity_scores(text)['compound']
        if score > 0.05:
            sentiment = "Positive "
        elif score < -0.05:
            sentiment = "Negative "
        else:
            sentiment = "Neutral "

        self.result_label.setText(f"Sentiment: {sentiment}\nScore: {score:.3f}")

    def analyze_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if not file_path:
            return

        try:
            df = pd.read_csv(file_path)
            if "review_text" not in df.columns:
                QMessageBox.critical(self, "Error", "CSV must have a 'review_text' column.")
                return

            def analyze_sentiment(text):
                score = sia.polarity_scores(str(text))['compound']
                if score > 0.05:
                    sentiment = 'Positive'
                elif score < -0.05:
                    sentiment = 'Negative'
                else:
                    sentiment = 'Neutral'
                return sentiment, score
            # aply sentiment analysis to each row in 'review_text'  
            
            
            # analyze_sentiment() returns (sentiment, score), zip(*) splits that into two new columns  
            df['sentiment'], df['compound_score'] = zip(*df['review_text'].apply(analyze_sentiment))

            # count how many positive, negative, and neutral sentiments there are, then turn into string  
            summary = df['sentiment'].value_counts().to_string()
            QMessageBox.information(self, "CSV Analysis Complete", f"Summary:\n\n{summary}")

            # create a new filename by adding "_with_sentiment" before the .csv extension  
            output_path = file_path.replace(".csv", "_with_sentiment.csv")

            # save the updated DataFrame (with sentiment results) to the new CSV file  
            df.to_csv(output_path, index=False)

            # show popup confirming where the file was saved  
            QMessageBox.information(self, "Saved", f"Results saved to:\n{output_path}")

        except Exception as e:
            # show error popup if something goes wrong (e.g., bad file, missing column)  
            QMessageBox.critical(self, "Error", f"Failed to process file:\n{e}")


            
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SentimentApp()
    window.show()
    sys.exit(app.exec())
