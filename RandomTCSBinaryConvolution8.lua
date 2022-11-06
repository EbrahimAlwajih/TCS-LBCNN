-- RandomTCSBinaryConvolution.lua

local THNN = require 'nn.THNN'
local RandomTCSBinaryConvolution8, parent = torch.class('cudnn.RandomTCSBinaryConvolution8', 'cudnn.SpatialConvolution')

function RandomTCSBinaryConvolution8:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
   parent.__init(self, nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
   self:reset()
end

function RandomTCSBinaryConvolution8:reset()
	local numElements = self.nInputPlane*self.nOutputPlane*self.kW*self.kH
	self.weight = torch.CudaTensor(self.nOutputPlane,self.nInputPlane,self.kW,self.kH):fill(0)
	--print('nInputPlane',self.nInputPlane)
	--print('nOutputPlane',self.nOutputPlane)
	--self.weight[{{},{},{2},{2}}]=-1
        self.weight = torch.reshape(self.weight,self.nOutputPlane,self.nInputPlane,self.kW*self.kH)
	local threshold={0.5}
	local index1=torch.Tensor({1,2,3,4,6,7,8,9})
	--local index=shuffle (index1)
	local i=1
	for nInputPlane = 1,self.nInputPlane do
		local index=shuffle (index1) -- for only 4 randome anchore weights 
		for nOutputPlane = 1,self.nOutputPlane do
			math.randomseed(os.clock())
                        threshold_idx=math.random(1)
			local rand1=math.random(1,8)
--print ('rand1',rand1, 'index[rand1]',index[rand1])
			self.weight[{{nOutputPlane},{nInputPlane},{index[rand1]}}]=threshold[threshold_idx]
			self.weight[{{nOutputPlane},{nInputPlane},{10-index[rand1]}}]=-self.weight[{{nOutputPlane},{nInputPlane},{index[rand1]}}]
			i=i+1

		end
	
	end
	self.weight = torch.reshape(self.weight,self.nOutputPlane,self.nInputPlane,self.kW,self.kH)
	--print(self.weight)
	self.bias = nil
	self.gradBias = nil	
	self.gradWeight = torch.CudaTensor(self.nOutputPlane, self.nInputPlane, self.kH, self.kW):fill(0) 	
end

function RandomTCSBinaryConvolution8:accGradParameters(input, gradOutput, scale)
end

function RandomTCSBinaryConvolution8:updateParameters(learningRate)
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

