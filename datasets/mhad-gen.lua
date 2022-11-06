-- mnist-gen.lua

local M = {}

local function convertToTensor(files)
   local data, labels

   for _, file in ipairs(files) do      
      local f = torch.load(file)      
      data = f.data:type(torch.getdefaulttensortype())
      labels = f.labels
   end

   return {
      data = data:contiguous():view(-1, 1, 32, 32),
      labels = labels,
   }
end

function M.exec(opt, cacheFile)
print (   opt.data .. '/mhad/train_32x32.t7')
   print(" | combining dataset into a single file")
   local trainData = convertToTensor({
      opt.data .. '/mhad/train_32x32.t7',
   })

   local testData = convertToTensor({
      opt.data .. '/mhad/test_32x32.t7',
   })

   print(" | saving MHAD dataset to " .. cacheFile)
   torch.save(cacheFile, {
      train = trainData,
      val = testData,
   })
end

return M
