require 'dp'
require 'cutorch'
require 'optim'
require 'image'
require 'cunn'
require 'os'

-------------------------- Parameters -------------------------------
-- window size
ws = 51
-- batch size for GPU
batch_size = 1500
-- number of classes
classes = 3
-- name of directory containing the test image
image_dir = '/home/sanuj/Downloads/cpctr_predict'
-- name of the test image
image_name = 'PrognosisTMABlock1_H_1_3_H_E_norm_crop.png'
-- model that has to be loaded
xp = torch.load('/home/sanuj/Projects/models/train_701_val_734.dat')
----------------------------------------------------------------------

input_image = image.load(image_dir .. '/' .. image_name, 3, 'byte')
channels = (#input_image)[1]; w = (#input_image)[2]; h = (#input_image)[3]

p = ws-1
module = nn.SpatialReflectionPadding(p/2, p/2, p/2, p/2)
module:cuda()
im = module:forward(input_image:cuda())
im = im:byte()
h = h+p
w = w+p

os.execute("mkdir " .. image_dir .. '/' .. 'tmp')
model = xp:model()

cropped = torch.Tensor(batch_size, channels, ws, ws):byte()
labels = torch.Tensor((h-ws+1)*(w-ws+1), classes)

counter = 0
last_counter = 1

for x = 0, h-ws do
	for y = 0, w-ws do
		print('Counter: ' .. counter .. ' cropped: ' .. (counter % batch_size)+1)
		cropped[{ {(counter % batch_size)+1}, {}, {}, {} }] = image.crop(im, x, y, x+ws, y+ws)
		if (counter+1) % batch_size == 0 then
			print('PREDICTING!!!')
			temp = model:forward(cropped[{ {1, batch_size}, {}, {}, {} }]):exp()
			labels[{ {(counter+1)-batch_size+1, counter+1}, {} }] = temp:double()
			last_counter = counter
		end
		counter = counter + 1
	end
end

if last_counter ~= (counter - 1) then
	temp = model:forward(cropped[{ {1, counter % batch_size}, {}, {}, {} }]):exp()
	labels[{ {last_counter+2, counter}, {} }] = temp:double()
end

for i = 1, channels do
	image.save(image_dir .. '/tmp/' .. i .. '.png', image.vflip(torch.reshape(labels[{ {}, {i} }], h-ws+1, w-ws+1)))
end
