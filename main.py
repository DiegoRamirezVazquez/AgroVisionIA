import customtkinter as ctk

from frontend.ui.interface import AgroVisionUI


def main():

    root = ctk.CTk()

    app = AgroVisionUI(root)

    root.mainloop()


if __name__ == '__main__':
    main()