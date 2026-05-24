import tkinter as tk
import cv2
from PIL import Image, ImageTk
import random

# Assuming you have a function to predict the hand sign from the webcam frame
def predict_hand_sign(frame):
    # Replace this with your actual model prediction code
    # Example: return model.predict(frame)
    predicted_class = random.choice(["rock", "paper", "scissors"])
    precision_rate = random.uniform(0.7, 0.9)
    return predicted_class, precision_rate

class Game:
    def __init__(self, master):
        self.master = master
        self.master.title("Rock Paper Scissors Game")

        self.choices = ["Rock", "Paper", "Scissors"]

        # Player choice
        self.player_choice = tk.StringVar()
        self.player_choice.set("Rock")

        # Computer choice
        self.computer_choice = tk.StringVar()

        # Result
        self.result_var = tk.StringVar()

        # Create GUI elements
        self.create_gui_elements()

        # Initialize webcam
        self.video_source = 0  # Use the default camera (change if needed)
        self.vid = cv2.VideoCapture(self.video_source)
        self.update()

    def create_gui_elements(self):
        # Player choice menu
        choice_menu = tk.OptionMenu(self.master, self.player_choice, *self.choices)
        choice_menu.pack(pady=10)

        # Play button
        play_button = tk.Button(self.master, text="Play", command=self.play_game)
        play_button.pack(pady=10)

        # Computer choice label
        computer_choice_label = tk.Label(self.master, text="Computer's Choice:")
        computer_choice_label.pack()

        # Result label
        result_label = tk.Label(self.master, textvariable=self.result_var)
        result_label.pack(pady=10)

        # Webcam display
        self.canvas = tk.Canvas(self.master, width=640, height=480)
        self.canvas.pack()

    def update(self):
        # Get a frame from the webcam
        ret, frame = self.vid.read()

        if ret:
            # Display the frame in the GUI
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        # Schedule the next update
        self.master.after(10, self.update)

    def play_game(self):
        # Get player choice
        player_choice = self.player_choice.get()

        # Get a frame from the webcam for computer choice prediction
        ret, frame = self.vid.read()

        if ret:
            # Predict computer choice and precision rate
            computer_choice, precision_rate = predict_hand_sign(frame)
            self.computer_choice.set(computer_choice)

            # Update result label
            self.result_var.set(f"Player chose {player_choice}. Computer chose {computer_choice} with {precision_rate:.2%} precision.")

    def __del__(self):
        # Release the webcam when the object is deleted
        if self.vid.isOpened():
            self.vid.release()

def main():
    root = tk.Tk()
    game = Game(root)
    root.mainloop()

if __name__ == "__main__":
    main()
