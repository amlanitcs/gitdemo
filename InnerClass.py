class Car:
    def __init__(selfy,Brand,Fuel,Transmission):
        selfy.brand= Brand
        selfy.fuel= Fuel
        selfy.transmission=Transmission
        # selfy.mil=selfy.milage(100,5).Mil()
        print(("how much kms ran"))
        M=float(input())
        P= print("how much petrol")
        P=float(input())
        selfy.mil = selfy.milage(M,P).Mil()

    def ShowDetails(self):
        print(self.brand + " runs on " + self.fuel + " and it has "+ self.transmission +" transmission.")


    class milage:
        def __init__(self, Km, Lt):
            self.Kilometer=Km
            self.Liter=Lt

        def Mil(self):
            print(self.Kilometer/self.Liter)





print("Please enter you Car Brand")
X= input()

print("please enter your fuel type")
Y= input()

print("please enter transmission type")
Z= input()

car2= Car(X,Y,Z)

car2.ShowDetails()
