import re #regular expression A RegEx, or Regular Expression, is a sequence of characters that forms a search pattern.
#RegEx can be used to check if a string contains the specified search pattern.
import math #math package
from collections import deque #used for queue purpose Deque can be implemented in python using the module “collections“. Deque is preferred over list in the cases 
#where we need quicker append and pop operations from both the ends of container
#Operations on deque :
#append() :- This function is used to insert the value in its argument to the right end of deque.
#appendleft() :- This function is used to insert the value in its argument to the left end of deque.
#pop() :- This function is used to delete an argument from the right end of deque.
#popleft() :- This function is used to delete an argument from the left end of deque.
"""
x is examples in training set
y is set of attributes
label is target attributes
Node is a class which has properties values, childs, and next
root is top node in the decision tree
"""

class Node(object): #make class for nodes values
    def __init__(self):#self aik object hai or _init_ aik constuctor __init__ is the constructor for a class.
        # The self parameter refers to the instance of the object
        self.value = None#nodes properties value initial is none same for others...
        self.next = None
        self.childs = None


# Simple class of Decision Tree
# Aimed for who want to learn Decision Tree, so it is not optimized
#make class of decision tree and used constructor and object with properties of deciosn class and value are sometime none
# becuase we furtherr used it
class DecisionTree(object):
    def __init__(self, sample, attributes, labels):
        self.sample = sample
        self.attributes = attributes
        self.labels = labels
        self.labelCodes = None
        self.labelCodesCount = None
        self.initLabelCodes()

        # print(self.labelCodes)

        self.root = None
        self.entropy = self.getEntropy([x for x in range(len(self.labels))])#here make entrophy object to get entrophy

    def initLabelCodes(self):
        self.labelCodes = []#make empty List later on used it as lable codes count
        self.labelCodesCount = []
        for lb in self.labels: #for loop for the append lable code in the array
            if lb not in self.labelCodes: #agr lb lable me se nahi hai then append krdo lb ko object of lable codes me
                self.labelCodes.append(lb)
                self.labelCodesCount.append(0)#and lable code count me value range zero krdo
            self.labelCodesCount[self.labelCodes.index(lb)] += 1

    def getLabelCodeId(self, sampleId):#function to get lable code ID with parameter of sample id
        return self.labelCodes.index(self.labels[sampleId])#it will return the sample lable code after matching the sample id

    def getAttributeValues(self, sampleIds, attributeId): #function to get attribute value with parametr of sample id and attribute id
        vals = []#make empty lsit and then add elemt at the end of list used the append func.
        for sid in sampleIds:
            val = self.sample[sid][attributeId]
            if val not in vals:
                vals.append(val)

        # print(vals)
        return vals
    def getEntropy(self, sampleIds):#function to get entrophy of
        entropy = 0
        labelCount = [0] * len(self.labelCodes)
        for sid in sampleIds:
            labelCount[self.getLabelCodeId(sid)] += 1

        # print("-ge", labelCount)

        for lv in labelCount:
            # print(lv)
            if lv != 0: #if yes then by using formula calculate Entrophy(math.log and sample ids length)
                # value must be greater than 1 otherwise it generate error
                entropy += -lv / len(sampleIds) * math.log(lv / len(sampleIds),   ) #-lv shows false values and lv is positivie
            else: #if it is not true then entropy val is zero
                entropy += 0
        return entropy

    def getDominantLabel(self, sampleIds): #func to get the dominant value lable
        labelCodesCount = [0] * len(self.labelCodes)
        for sid in sampleIds:
            labelCodesCount[self.labelCodes.index(self.labels[sid])] += 1
        return self.labelCodes[labelCodesCount.index(max(labelCodesCount))] #return the max lable val in the index of lable code count

    def getInformationGain(self, sampleIds, attributeId):#function to get the information gain
        gain = self.getEntropy(sampleIds)#take value of entrophy
        attributeVals = []#make empty list of attr val and val counts
        attributeValsCount = []
        attributeValsIds = []
        for sid in sampleIds:
            val = self.sample[sid][attributeId]
            if val not in attributeVals:
                attributeVals.append(val)
                attributeValsCount.append(0)
                attributeValsIds.append([])#used append funct here to append the attribute val in the empty list
            vid = attributeVals.index(val)
            attributeValsCount[vid] += 1 #attrvalcou=attrvalcou+1
            attributeValsIds[vid].append(sid)
        # print("-gig", self.attributes[attributeId])
#zip function returns a zip object, which is an iterator of tuples
# where the first item in each passed iterator is paired together, and then the second item in each passed
# iterator are paired together etc.
#If the passed iterators have different lengths, the iterator with the least items decides the length of the new iterator.
        for vc, vids in zip(attributeValsCount, attributeValsIds):#atribute val coun and id are the iterator in zip func
            # print("-gig", vids)
            gain -= vc / len(sampleIds) * self.getEntropy(vids) #to get information gain value
        return gain

    def getAttributeMaxInformationGain(self, sampleIds, attributeIds): #func to get max information gain value to make it as a root
        attributesEntropy = [0] * len(attributeIds)
        for i, attId in zip(range(len(attributeIds)), attributeIds):
            attributesEntropy[i] = self.getInformationGain(sampleIds, attId)
        maxId = attributeIds[attributesEntropy.index(max(attributesEntropy))]
        return self.attributes[maxId], maxId#max entrophy return

    def isSingleLabeled(self, sampleIds): #if one single lable and value is false then it will simply generte false like rest of world
        # otherwise return true in example
        label = self.labels[sampleIds[0]]
        for sid in sampleIds:
            if self.labels[sid] != label:
                return False

        return True

    def getLabel(self, sampleId):#func to get label with sample id
        return self.labels[sampleId]

    def id3(self): #make id3 fuc with call the above object self
        sampleIds = [x for x in range(len(self.sample))]
        attributeIds = [x for x in range(len(self.attributes))]
        self.root = self.id3Recv(sampleIds, attributeIds, self.root)

    def id3Recv(self, sampleIds, attributeIds, root): #func which return root val
        root = Node()  # Initialize current root
        if self.isSingleLabeled(sampleIds):
            root.value = self.labels[sampleIds[0]] #at 0 index
            return root

        # print(attributeIds)

        if len(attributeIds) == 0: #if attId length is equal to zero then most dominant label will be store in root val
            root.value = self.getDominantLabel(sampleIds)
            return root
        bestAttrName, bestAttrId = self.getAttributeMaxInformationGain(
            sampleIds, attributeIds)

        # print(bestAttrName)
        root.value = bestAttrName #best attr name is equal to the root val
        root.childs = []  # Create list of children
        for value in self.getAttributeValues(sampleIds, bestAttrId):
            # print(value)
            child = Node() #child is equal to node class
            child.value = value #put the val of node in child val
            root.childs.append(child)  # Append new child node to current
            # root
            childSampleIds = [] #make empty list of childsample id
            for sid in sampleIds:
                if self.sample[sid][bestAttrId] == value:
                    childSampleIds.append(sid) #append it 
            if len(childSampleIds) == 0: #if len of child sampleid is equal to zero then store the dominant val in child next attr
                child.next = self.getDominantLabel(sampleIds)
            else:
                # print(bestAttrName, bestAttrId)

                # print(attributeIds)
                if len(attributeIds) > 0 and bestAttrId in attributeIds:
                    toRemove = attributeIds.index(bestAttrId)
                    attributeIds.pop(toRemove) #for detail of pop function go to the above dequue operations
                child.next = self.id3Recv(
                    childSampleIds, attributeIds, child.next)
        return root
    def printTree(self): #func to print tree
        if self.root: #if object.root then deque root and append the root in roots var
            roots = deque()
            roots.append(self.root)
            while len(roots) > 0: #if len of root is graeter then zero then pop(stack) the valof root left
                root = roots.popleft()# go to deque detail 
                print(root.value) #print root val
                if root.childs: #for child same procedure and next also
                    for child in root.childs:
                        print('({})'.format(child.value)) #format() is one of the string formatting methods which allows multiple substitutions and value formatting. 
						#This method concatenate elements within a string through positional formatting.
                        roots.append(child.next)
                elif root.next:#if not child then go to root next
                    print(root.next)
def test(): #func used as test
    f = open('TESTER.csv') #open the data set with csv extension
    attributes = f.readline().split(',')
    attributes = attributes[1:len(attributes) - 1]
    print(attributes)
    sample = f.readlines()
    f.close()
    for i in range(len(sample)):
        sample[i] = re.sub('\d+,', '', sample[i])#used re \d+ means one or more digit [0-9]
        sample[i] = sample[i].strip().split(',')
    labels = []#empty list of label
    for s in sample:
        labels.append(s.pop()) #append the pop val(deque) in labels
    print("Sample DATA: \n",sample)#print sample data
    print("Lable Data: \n",labels)#print lable data
    decisionTree = DecisionTree(sample, attributes, labels)#sample,attr,lables of dataset
    print("\n System entropy {}".format(decisionTree.entropy))#shows the entrophy
    decisionTree.id3()#call the class with fun id3
    decisionTree.printTree()#to print the decison tree


if __name__ == '__main__':
    test() #call the test func
