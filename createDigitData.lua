-- main.lua

require 'torch'
require 'math'
require 'image'

local matio = require 'matio'
local MHAD = matio.load('/media/eqw/New Volume/lbcnn/Datasets/mhad/MHADTest.mat')
nData=10000
local MHADdata = MHAD.data:type(torch.getdefaulttensortype())
local MHADlabels=MHAD.label
print('Type of labels in MHADlabels: ',MHADlabels:type())
print('Type of data in MHADdata: ',MHADdata:type())


local f = torch.load('/media/eqw/New Volume/lbcnn/Datasets/mnist/test_32x32.t7','ascii')
 print(f)
local MNISTdata = f.data:type(torch.getdefaulttensortype())
local MNISTlabels = f.labels
print('Type of labels in mnist: ',MNISTlabels:type())
print('Type of data in mnist: ',MNISTdata:type())

local data=torch.cat(MNISTdata,MHADdata,1)
local labels= torch.ByteTensor(2*nData):fill(0)
for i=1,nData do
	labels[i]=MNISTlabels[i]
end 

for i=1,nData do
--print(testlabels[{1,i}])
--print(bilinguallabel:type())
--print(testlabels:type())
	labels[nData+i]= torch.ByteTensor(1):fill(MHADlabels[{1,i}])
	if labels[nData+i]==0 then
		labels[nData+i]=11
	elseif labels[nData+i]==1 then
		labels[nData+i]=2
	elseif labels[nData+i]==2 then
		labels[nData+i]=12
	elseif labels[nData+i]==3 then
		labels[nData+i]=13
	elseif labels[nData+i]==4 then
		labels[nData+i]=14
	elseif labels[nData+i]==5 then
		labels[nData+i]=1
	elseif labels[nData+i]==6 then
		labels[nData+i]=8
	elseif labels[nData+i]==7 then
		labels[nData+i]=15
	elseif labels[nData+i]==8 then
		labels[nData+i]=16
	elseif labels[nData+i]==9 then
		labels[nData+i]=10
	end
end 

print('Type of bilinguallabel: ',labels:type())
print('Type of bilingualdata: ',data:type())

--Create the table to save
Data_to_Write = { data = data, labels = labels }

--Save the table in the /tmp
         -- torch.save("/media/eqw/New Volume/lbcnn/Datasets/dbilingual/test_32x32.t7", Data_to_Write)

 testt7 = torch.load('/media/eqw/New Volume/lbcnn/Datasets/dbilingual/test_32x32.t7')
 print(testt7)


for i=1, 0 do
--print (table.getn(data))
--print(data[{{i},{1},{},{}}])
--print(data:size())
x=data[{{10000+i},{1},{},{}}]
ou=image.toDisplayTensor{x, padding=2, zoom=4}; image.save(tostring(labels[10000+i])..'.png', ou)
--  image.display(x)

end

