-- RandomCSBinaryConvolution.lua

local THNN = require 'nn.THNN'
local RandomCSBinaryConvolution8, parent = torch.class('cudnn.RandomCSBinaryConvolution8', 'cudnn.SpatialConvolution')

function RandomCSBinaryConvolution8:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
   parent.__init(self, nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
   self:reset()
end

function RandomCSBinaryConvolution8:reset()
	local numElements = self.nInputPlane*self.nOutputPlane*self.kW*self.kH
	self.weight = torch.CudaTensor(self.nOutputPlane,self.nInputPlane,self.kW,self.kH):fill(0)
	--print('nInputPlane',self.nInputPlane)
	--print('nOutputPlane',self.nOutputPlane)
	--self.weight[{{},{},{2},{2}}]=-1
        self.weight = torch.reshape(self.weight,self.nOutputPlane,self.nInputPlane,self.kW*self.kH)

	local index1=torch.Tensor({1,2,3,4,6,7,8,9})
	--local index=shuffle (index1)
	local i=1
	for nInputPlane = 1,self.nInputPlane do
		local index=shuffle (index1) -- for only 4 randome anchore weights 
		for nOutputPlane = 1,self.nOutputPlane do
			math.randomseed(os.clock())
			local rand1=math.random(1,8)
--print ('rand1',rand1, 'index[rand1]',index[rand1])
			self.weight[{{nOutputPlane},{nInputPlane},{index[rand1]}}]=1
			self.weight[{{nOutputPlane},{nInputPlane},{10-index[rand1]}}]=-1
			i=i+1

		end
	
	end
	self.weight = torch.reshape(self.weight,self.nOutputPlane,self.nInputPlane,self.kW,self.kH)
	--print(self.weight)
	self.bias = nil
	self.gradBias = nil	
	self.gradWeight = torch.CudaTensor(self.nOutputPlane, self.nInputPlane, self.kH, self.kW):fill(0) 	
end

function RandomCSBinaryConvolution8:accGradParameters(input, gradOutput, scale)
end

function RandomCSBinaryConvolution8:updateParameters(learningRate)
end

function shuffle (arr)
	size=arr:numel()
	for i=1, size do
		math.randomseed(os.time())
		local rand1=math.random(size)
		arr[i],arr[rand1] = arr[rand1], arr[i]
	end
	return arr
end

