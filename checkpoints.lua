-- checkpoints.lua
--[[
]]--
local RandomBinaryConvolution, parent
local function load_lbcnn_based_layer(opt)
   local latestPath = paths.concat(opt.resume, 'latest.t7')
   if paths.filep(latestPath) then
       if opt.netType == 'resnet-tcsbinary-felix' then
           RandomBinaryConvolution, parent = torch.class('cudnn.RandomTCSBinaryConvolution', 'cudnn.SpatialConvolution')
       elseif opt.netType == 'resnet-csbinary-felix' then
           RandomBinaryConvolution, parent = torch.class('cudnn.RandomCSBinaryConvolution', 'cudnn.SpatialConvolution')
       elseif opt.netType == 'resnet-binary-felix' then
           RandomBinaryConvolution, parent = torch.class('cudnn.RandomBinaryConvolution', 'cudnn.SpatialConvolution')
       elseif opt.netType == 'resnet-csbinary-felix8' then
           RandomBinaryConvolution, parent = torch.class('cudnn.RandomCSBinaryConvolution8', 'cudnn.SpatialConvolution')
       elseif opt.netType == 'resnet-binary-felix8' then
           RandomBinaryConvolution, parent = torch.class('cudnn.RandomBinaryConvolution8', 'cudnn.SpatialConvolution')
       elseif opt.netType == 'resnet-tcsbinary-felix8' then
           RandomBinaryConvolution, parent = torch.class('cudnn.RandomTCSBinaryConvolution8', 'cudnn.SpatialConvolution')
       else 
           return nil
       end
    end
    return RandomBinaryConvolution, parent
end

require 'lfs'
local checkpoint = {}
RandomBinaryConvolution, parent = torch.class('cudnn.RandomTCSBinaryConvolution', 'cudnn.SpatialConvolution')
local function deepCopy(tbl)
   -- creates a copy of a network with new modules and the same tensors
   local copy = {}
   for k, v in pairs(tbl) do
      if type(v) == 'table' then
         copy[k] = deepCopy(v)
      else
         copy[k] = v
      end
   end
   if torch.typename(tbl) then
      torch.setmetatable(copy, torch.typename(tbl))
   end
   return copy
end

function checkpoint.latest(opt)
   if opt.resume == 'none' then
      return nil
   end
   local latestPath = paths.concat(opt.resume, 'latest.t7')
   if not paths.filep(latestPath) then
      return nil
   end
   print('=> Loading checkpoint ' .. latestPath)
   --RandomBinaryConvolution, parent=load_lbcnn_based_layer(opt)
   local latest = torch.load(latestPath)
   local optimState = torch.load(paths.concat(opt.resume, latest.optimFile))
   return latest, optimState
end

function checkpoint.best(opt)
   if opt.resume == 'none' then
      return nil
   end
   local latestPath = paths.concat(opt.resume, 'latest.t7')
   if not paths.filep(latestPath) then
      return nil
   end
   print('=> Loading checkpoint ' .. latestPath)
   --RandomBinaryConvolution, parent=load_lbcnn_based_layer(opt)
   local latest = torch.load(latestPath)
   local optimState = torch.load(paths.concat(opt.resume, latest.optimFile))
   return latest, optimState
end


function checkpoint.save(opt, epoch, model, optimState, bestModel)
   -- Don't save the DataParallelTable for easier loading on other machines
   if torch.type(model) == 'nn.DataParallelTable' then
      model = model:get(1)
   end
   model = deepCopy(model):float():clearState()

   local modelFile = paths.concat(opt.save, 'model_' .. epoch .. '.t7')
   local optimFile = paths.concat(opt.save, 'optimState_' .. epoch .. '.t7')
   
   torch.save(paths.concat(opt.save, modelFile), model)
   torch.save(paths.concat(opt.save, optimFile), optimState)
   torch.save(paths.concat(opt.save, 'latest.t7'), {
      epoch = epoch,
      modelFile = modelFile,
      optimFile = optimFile,
   })

   print("Deleting old models from disk")
   local modelFile = paths.concat(opt.save, 'model_' .. (epoch-2) .. '.t7')
   local optimFile = paths.concat(opt.save, 'optimState_' .. (epoch-2) .. '.t7')

   if lfs.attributes(modelFile) then
   	os.remove(modelFile)
   end
   if lfs.attributes(optimFile) then
   	os.remove(optimFile)
   end

   print("Saving model to disk")
   if bestModel then
	local modelFile = paths.concat(opt.save, 'model_best.t7')
   	local optimFile = paths.concat(opt.save, 'optimState_best.t7')
   
   	torch.save(paths.concat(opt.save, modelFile), model)
   	torch.save(paths.concat(opt.save, optimFile), optimState)
   	torch.save(paths.concat(opt.save, 'best.t7'), {
    	  epoch = epoch,
      	  modelFile = modelFile,
          optimFile = optimFile,
   })

      --torch.save(paths.concat(opt.save, 'model_best.t7'), model)
   end

end

return checkpoint
