import datetime


class Cars:

    Material = "Steel"

    def __init__(self,HowmanyWheels,HowOld):
        self.age= int(HowOld)
        self.tyresCount= int(HowmanyWheels)

    def showAge(Any):
        return Any.age

    @classmethod
    def ChangeMaterial(clsa): #even in class methid arguments, the default parameter can be of any name
        #clsa.Material = "Aluminium"
        return clsa.Material

    def changeMaterial(whatever):
        whatever.Material = "Gold"
        return  whatever.Material

    def getExpiryChecked(any):
        if any.age > 15:
            return ("Expired.")
        else:
            return ("Not Expired.")


    def getWheelCount(self):
        if self.tyresCount == 3:
            return ("Auto")
        elif self.tyresCount == 4:
            return ("car")
        else:
            return ("truck")

print("enter the number of tyres of your vehicle:")
x=int(input())

print("enter the year of purchase:")
y=int(input())

CustomerVehicle = Cars(x,y)

print("your vehicle is a " + CustomerVehicle.getWheelCount() + ". And it's expiry status is " + CustomerVehicle.getExpiryChecked())
print("your class's object is allocated the address in the heap memory: " + str(id(CustomerVehicle)))
print(CustomerVehicle.showAge())

print(CustomerVehicle.changeMaterial())
print(CustomerVehicle.Material)

print(Cars.ChangeMaterial())
anotherCar = Cars(4, 18)
print(anotherCar.Material)
print(anotherCar.changeMaterial())



