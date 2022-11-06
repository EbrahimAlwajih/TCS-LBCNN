-- RandomCSBinaryConvolution.lua

local THNN = require 'nn.THNN'
local RandomCSBinaryConvolution, parent = torch.class('cudnn.RandomCSBinaryConvolution', 'cudnn.SpatialConvolution')

function RandomCSBinaryConvolution:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
   parent.__init(self, nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
   self:reset()
end

function RandomCSBinaryConvolution:reset()
	local numElements = self.nInputPlane*self.nOutputPlane*self.kW*self.kH
	self.weight = torch.CudaTensor(self.nOutputPlane,self.nInputPlane,self.kW,self.kH):fill(0)
	self.weight = torch.reshape(self.weight,numElements)
	local index = torch.Tensor(torch.floor(kSparsity*numElements)):random(numElements)
	
	for i = 1,index:numel() do
		local modd= index[i] % 9 
		if modd==0 then modd=9 end
		if modd<5 then
			if self.weight[index[i]+(5-modd)*2] == 0 then
				--print('moda < 5')
				self.weight[index[i]]=torch.bernoulli(0.5)*2-1
				self.weight[index[i]+(5-modd)*2]=-self.weight[index[i]]
			end
		else if modd > 5 then 
			if self.weight[index[i]-(modd-5)*2] == 0 then
				--print('moda > 5')
				self.weight[index[i]]=torch.bernoulli(0.5)*2-1
				self.weight[index[i]-(modd-5)*2]=-self.weight[index[i]]
			end
		end
		end
		--self.weight[index[i]] = torch.bernoulli(0.5)*2-1
	end
	self.weight = torch.reshape(self.weight,self.nOutputPlane,self.nInputPlane,self.kW,self.kH)
        --print(self.weight)
	--print ( 'size', index:size())	
	self.bias = nil
	self.gradBias = nil	
	self.gradWeight = torch.CudaTensor(self.nOutputPlane, self.nInputPlane, self.kH, self.kW):fill(0)	


	
	
end


function RandomCSBinaryConvolution:accGradParameters(input, gradOutput, scale)
end

function RandomCSBinaryConvolution:updateParameters(learningRate)
end

function shuffle (arr)
	size=arr:numel()
	for i=1, size do
		local rand1=math.random(size)
		arr[i],arr[rand1] = arr[rand1], arr[i]
	end
	return arr
end

