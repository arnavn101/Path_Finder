
import tkinter
from functools import partial

def tkinterCall_result(self,label_result, n1, n2):
    num1 = (n1.get())
    num2 = (n2.get())
    result = int(num1)+int(num2)
    label_result.config(text="Result is %d" % result)
    return
root = tkinter.Tk()
root.geometry('400x200+100+200')
root.title("Veevek's Awesome Calulator")
numero1 = tkinter.StringVar()
numero2 = tkinter.StringVar()
tkinterTitle_label = tkinter.Label(root, text="Built by Veevek").grid(row=0, column=2)
tkinterNumber1_label = tkinter.Label(root, text="Enter one number").grid(row=1, column=0)
tkinterNumber2_label = tkinter.Label(root, text="Enter one more number").grid(row=2, column=0)
tkinterResult = tkinter.Label(root)
tkinterResult.grid(row=7, column=2)
tkinterNumber1 = tkinter.Entry(root, textvariable=numero1).grid(row=1, column=2)
tkinterNumber2 = tkinter.Entry(root, textvariable=numero2).grid(row=2, column=2)
tkinterCall_result = partial(tkinterCall_result, tkinterResult, numero1, numero2)
buttonCal = tkinter.Button(root, text="Calculate", command=tkinterCall_result).grid(row=3, column=0)
root.mainloop()

