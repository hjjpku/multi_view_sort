import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import pickle
import os
from tensorboardX import SummaryWriter
import time
from collections import Iterable
from .NEM_utilities import compute_prior, compute_outer_loss, clip_gradient

class ModelNetTrainer(object):

    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, \
                 model_name, log_dir, num_views=12, class_num=40):
        self.class_num = class_num
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.model_name = model_name
        self.log_dir = log_dir
        self.num_views = num_views
        self.gpu_num = torch.cuda.device_count()
        if isinstance(model, Iterable):
            self.model = []
            for i in range(len(model)):
                model[i].cuda()
                self.model.append(torch.nn.DataParallel(model[i]))
            self.model = tuple(self.model)
        else:
            self.model = model
            self.model.cuda()
        if self.log_dir is not None:
            self.writer = SummaryWriter(log_dir)


    def train(self, n_epochs):

        best_acc = 0
        i_acc = 0
        self.model.train()
        for epoch in range(n_epochs):
            # permute data for mvcnn
            rand_idx = np.random.permutation(int(len(self.train_loader.dataset.filepaths)/self.num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.train_loader.dataset.filepaths[rand_idx[i]*self.num_views:(rand_idx[i]+1)*self.num_views])
            self.train_loader.dataset.filepaths = filepaths_new

            # plot learning rate
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.writer.add_scalar('params/lr', lr, epoch)

            # train one epoch
            out_data = None
            in_data = None
            for i, data in enumerate(self.train_loader):

                if self.model_name == 'mvcnn':
                    N,V,C,H,W = data[1].size()
                    in_data = Variable(data[1]).view(-1,C,H,W).cuda()
                else:
                    in_data = Variable(data[1].cuda())
                target = Variable(data[0]).cuda().long()

                self.optimizer.zero_grad()

                out_data = self.model(in_data)

                loss = self.loss_fn(out_data, target)
                
                self.writer.add_scalar('train/train_loss', loss, i_acc+i+1)

                pred = torch.max(out_data, 1)[1]
                results = pred == target
                correct_points = torch.sum(results.long())

                acc = correct_points.float()/results.size()[0]
                self.writer.add_scalar('train/train_overall_acc', acc, i_acc+i+1)

                loss.backward()
                self.optimizer.step()
                
                log_str = 'epoch %d, step %d: train_loss %.3f; train_acc %.3f' % (epoch+1, i+1, loss, acc)
                if (i+1)%1==0:
                    print(log_str)
            i_acc += i

            # evaluation
            if (epoch+1)%1==0:
                with torch.no_grad():
                    loss, val_overall_acc, val_mean_class_acc = self.update_validation_accuracy(epoch)
                self.writer.add_scalar('val/val_mean_class_acc', val_mean_class_acc, epoch+1)
                self.writer.add_scalar('val/val_overall_acc', val_overall_acc, epoch+1)
                self.writer.add_scalar('val/val_loss', loss, epoch+1)

            # save best model
            if val_overall_acc > best_acc:
                best_acc = val_overall_acc
                self.model.save(self.log_dir, epoch)
 
            # adjust learning rate manually
            if epoch > 0 and (epoch+1) % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr']*0.5

        # export scalar data to JSON for external processing
        self.writer.export_scalars_to_json(self.log_dir+"/all_scalars.json")
        self.writer.close()


    def train_nem_mvcnn(self, n_epochs):

        nem, c_net = self.model
        class_nem = nem.module
        best_acc = 0
        i_acc = 0

        nem.train()
        c_net.train()
        iter_num = class_nem.iter_num
        type = class_nem.type
        prior = compute_prior(type) # EM prior
        stop_gamma_grad = class_nem.stop_gamma_grad
        if_gamma_prior = class_nem.if_gamma_prior
        if_decoder_warm_up = class_nem.if_decoder_warm_up
        if_total_loss = class_nem.if_total_loss
        for epoch in range(n_epochs):
            # plot learning rate

            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.writer.add_scalar('params/lr', lr, epoch)

            # train one epoch
            for i, i_data in enumerate(self.train_loader):

                start = time.time()

                # dataparallel check
                mod = i_data[0].shape[0] % self.gpu_num
                if mod == 0:
                    data = i_data
                else:
                    b = i_data[0].shape[0]
                    in_data_init = torch.zeros(b+(self.gpu_num-mod),i_data[1].shape[1], i_data[1].shape[2])
                    in_data_init[0:b] = i_data[1]
                    in_data_init[-(self.gpu_num-mod):] = i_data[1][-(self.gpu_num-mod):]
                    target_data_init = torch.zeros(b+(self.gpu_num-mod))
                    target_data_init[0:b] = i_data[0]
                    target_data_init[-(self.gpu_num-mod):] = i_data[0][-(self.gpu_num-mod):]
                    data = [target_data_init, in_data_init]



                in_data = data[1] # B,M,C,H,W or B,M,C

                # init states
                init_state = class_nem.init_state(in_data.shape,in_data)
                state = (init_state[0], init_state[1], init_state[2])
                gamma_prior = init_state[3]
                gamma = init_state[3]

                if len(in_data.size()) == 3:
                    in_data = in_data.unsqueeze(0).expand(iter_num,-1,-1,-1)
                elif len(in_data.size()) == 5:
                    in_data =in_data.unsqueeze(0).expand(iter_num,-1,-1,-1,-1,-1)
                else:
                    raise KeyError('wrong size for data')

                in_data = Variable(in_data).cuda().float()
                target = Variable(data[0]).cuda().long()

                self.optimizer.zero_grad()
                nem_losses, intra_losses, inter_losses, nem_outputs = [], [], [], []

                # run NEM
                for j in range(iter_num):
                    # em iteration
                    input = in_data[j].unsqueeze(1) # B,1,M...
                    #state = nem(input, state)
                    if if_decoder_warm_up == 1 and epoch < 10:
                        state = nem(input, (state[0],state[1],gamma_prior))
                    else:
                        state = nem(input,state)
                        #state = nem(input, (state[0], state[1], gamma_prior))

                    theta, pred_f, gamma =  state


                    # compute nem losses
                    nem_loss, intra_loss, inter_loss = compute_outer_loss(pred_f, gamma, input, prior, stop_gamma_grad, gamma_prior, if_gamma_prior)
                    nem_losses.append(nem_loss)
                    intra_losses.append(intra_loss)
                    inter_losses.append(inter_loss)
                    nem_outputs.append(state)

                out_data = c_net(nem_outputs[-1][0],gamma)
                loss_weight = 1
                closs = self.loss_fn(out_data, target)
                loss = (closs * loss_weight + nem_losses[-1]) if if_total_loss == 0 else (loss_weight * closs + torch.mean(torch.Tensor(nem_losses).cuda()))
                #loss = self.loss_fn(out_data, target)
                i_acc += 1
                self.writer.add_scalar('train/train_loss', loss, i_acc)
                self.writer.add_scalar('train/c_loss', closs, i_acc)
                self.writer.add_scalar('train/nem_loss', nem_losses[-1], i_acc)
                self.writer.add_scalar('train/intra_loss', intra_losses[-1], i_acc)
                self.writer.add_scalar('train/inter_loss', inter_losses[-1], i_acc)

                pred = torch.max(out_data, 1)[1]
                results = pred == target
                correct_points = torch.sum(results.long())

                acc = correct_points.float() / results.size()[0]
                self.writer.add_scalar('train/train_overall_acc', acc, i_acc)

                loss.backward()
                if class_nem.bn_init_sigma == 1:
                    clip_gradient(self.optimizer,5)
                self.optimizer.step()

                torch.cuda.synchronize()
                log_str = 'epoch %d, step %d: train_loss %.3f; train_acc %.3f. Runtime %.3f' % (epoch + 1, i + 1, loss, acc, time.time()-start)
                if (i + 1) % 1 == 0:
                    print(log_str)


            if epoch == 0 and class_nem.bn_init_sigma == 1:
                total_sigma = class_nem.total_sigma / i_acc
                torch.save(total_sigma,'total_sigma.pth')
            # evaluation
            print('run evaluation...')
            start = time.time()
            if (epoch + 1) % 1 == 0:
                with torch.no_grad():
                    loss, val_overall_acc, val_mean_class_acc = self.update_validation_accuracy_nem(epoch)
                self.writer.add_scalar('val/val_mean_class_acc', val_mean_class_acc, epoch + 1)
                self.writer.add_scalar('val/val_overall_acc', val_overall_acc, epoch + 1)
                self.writer.add_scalar('val/val_loss', loss, epoch + 1)
            torch.cuda.synchronize()
            print('evaluation ended... runtime: ', time.time()-start)

            # save best model
            if val_overall_acc > best_acc:
                best_acc = val_overall_acc
                nem.module.save(self.log_dir, epoch)
                c_net.module.save(self.log_dir, epoch)

            # adjust learning rate manually
            if epoch > 0 and (epoch + 1) % 15 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5

            # export scalar data to JSON for external processing
        self.writer.export_scalars_to_json(self.log_dir + "/all_scalars.json")
        self.writer.close()

    def update_validation_accuracy_nem(self, epoch):
        all_correct_points = 0
        all_points = 0

        # in_data = None
        # out_data = None
        # target = None

        wrong_class = np.zeros(self.class_num)
        samples_class = np.zeros(self.class_num)
        all_loss = 0

        nem, c_net = self.model
        nem.eval()
        c_net.eval()

        class_nem = nem.module # get NEM object
        cluster_n = class_nem.k
        iter_num = class_nem.iter_num
        type = class_nem.type
        prior = compute_prior(type)  # EM prior
        stop_gamma_grad = class_nem.stop_gamma_grad
        if_gamma_prior = class_nem.if_gamma_prior
        if_decoder_warm_up = class_nem.if_decoder_warm_up
        for _, i_data in enumerate(self.val_loader, 0):

            # data[0]: label; data[1]:input; data[2]:file_path
            # N, M, C, H, W = data[1].size()
            mod = i_data[0].shape[0] % self.gpu_num
            if mod == 0:
                data = i_data
            else:
                b = i_data[0].shape[0]
                in_data_init = torch.zeros(b + (self.gpu_num - mod), i_data[1].shape[1], i_data[1].shape[2])
                in_data_init[0:b] = i_data[1]
                in_data_init[-(self.gpu_num - mod):] = i_data[1][-(self.gpu_num - mod):]
                target_data_init = torch.zeros(b + (self.gpu_num - mod))
                target_data_init[0:b] = i_data[0]
                target_data_init[-(self.gpu_num - mod):] = i_data[0][-(self.gpu_num - mod):]
                data = [target_data_init, in_data_init]

            in_data = data[1]  # B,M,C,H,W or B,M,C

            # init states
            init_state = class_nem.init_state(in_data.shape, in_data)
            state = (init_state[0], init_state[1], init_state[2])
            gamma_prior = init_state[3]
            gamma = init_state[3]

            if len(in_data.size()) == 3:
                in_data = in_data.unsqueeze(0).expand(iter_num, -1, -1, -1)
            elif len(in_data.size()) == 5:
                in_data = in_data.unsqueeze(0).expand(iter_num, -1, -1, -1, -1, -1)
            else:
                raise KeyError('wrong size for data')

            in_data = Variable(in_data).cuda().float()
            target = Variable(data[0]).cuda()

            nem_losses, intra_losses, inter_losses, nem_outputs = [], [], [], []

            # run NEM
            for j in range(iter_num):
                # em iteration
                input = in_data[j].unsqueeze(1)  # B,1,M...
                # state = nem(input, state)
                state = nem(input, state)
                theta, pred_f, gamma = state

                # compute nem losses
                nem_loss, intra_loss, inter_loss = compute_outer_loss(pred_f, gamma, input, prior, stop_gamma_grad,gamma_prior, if_gamma_prior)
                nem_losses.append(nem_loss)
                intra_losses.append(intra_loss)
                inter_losses.append(inter_loss)
                nem_outputs.append(state)

            out_data = c_net(nem_outputs[-1][0],gamma)
            pred = torch.max(out_data, 1)[1]
            all_loss += self.loss_fn(out_data, target).cpu().data.numpy()
            results = pred == target

            for i in range(results.size()[0]):
                if not bool(results[i].cpu().data.numpy()):
                    wrong_class[target.cpu().data.numpy().astype('int')[i]] += 1
                samples_class[target.cpu().data.numpy().astype('int')[i]] += 1
            correct_points = torch.sum(results.long())

            all_correct_points += correct_points
            all_points += results.size()[0]

        print('Total # of test models: ', all_points)
        val_mean_class_acc = np.mean((samples_class - wrong_class) / samples_class)
        each_class_acc = (samples_class - wrong_class) / samples_class
        np.save(str(cluster_n)+'_each_class_acc.npy',each_class_acc)
        acc = float(all_correct_points) / all_points
        val_overall_acc = acc
        loss = all_loss / len(self.val_loader)

        print('val mean class acc. : ', val_mean_class_acc)
        print('val overall acc. : ', val_overall_acc)
        print('val loss : ', loss)

        nem.train()
        c_net.train()

        return loss, val_overall_acc, val_mean_class_acc

    def save_feature_nem(self, epoch):
        all_correct_points = 0
        all_points = 0

        # in_data = None
        # out_data = None
        # target = None

        wrong_class = np.zeros(self.class_num)
        samples_class = np.zeros(self.class_num)
        all_loss = 0

        nem, c_net = self.model
        nem.eval()
        c_net.eval()

        class_nem = nem.module # get NEM object

        iter_num = class_nem.iter_num
        type = class_nem.type
        prior = compute_prior(type)  # EM prior
        stop_gamma_grad = class_nem.stop_gamma_grad
        if_gamma_prior = class_nem.if_gamma_prior
        for _, i_data in enumerate(self.val_loader, 0):

            '''
            # data[0]: label; data[1]:input; data[2]:file_path
            # N, M, C, H, W = data[1].size()
            mod = i_data[0].shape[0] % self.gpu_num
            if mod == 0:
                data = i_data
            else:
                b = i_data[0].shape[0]
                in_data_init = torch.zeros(b + (self.gpu_num - mod), i_data[1].shape[1], i_data[1].shape[2])
                in_data_init[0:b] = i_data[1]
                in_data_init[-(self.gpu_num - mod):] = i_data[1][-(self.gpu_num - mod):]
                target_data_init = torch.zeros(b + (self.gpu_num - mod))
                target_data_init[0:b] = i_data[0]
                target_data_init[-(self.gpu_num - mod):] = i_data[0][-(self.gpu_num - mod):]
                data = [target_data_init, in_data_init]
            '''
            data = i_data

            in_data = data[1]  # B,M,C,H,W or B,M,C

            assert (in_data.shape[0] == 1), 'save data only support bs=1'

            # init states
            init_state = class_nem.init_state(in_data.shape, in_data)
            state = (init_state[0], init_state[1], init_state[2])
            gamma_prior = init_state[3]
            gamma = init_state[3]

            if len(in_data.size()) == 3:
                in_data = in_data.unsqueeze(0).expand(iter_num, -1, -1, -1)
            elif len(in_data.size()) == 5:
                in_data = in_data.unsqueeze(0).expand(iter_num, -1, -1, -1, -1, -1)
            else:
                raise KeyError('wrong size for data')

            in_data = Variable(in_data).cuda().float()
            target = Variable(data[0]).cuda()

            nem_losses, intra_losses, inter_losses, nem_outputs = [], [], [], []

            # run NEM
            for j in range(iter_num):
                # em iteration
                input = in_data[j].unsqueeze(1)  # B,1,M...
                state = nem(input, state)
                theta, pred_f, gamma = state

                # compute nem losses
                nem_loss, intra_loss, inter_loss = compute_outer_loss(pred_f, gamma, input, prior, stop_gamma_grad,gamma_prior, if_gamma_prior)
                nem_losses.append(nem_loss)
                intra_losses.append(intra_loss)
                inter_losses.append(inter_loss)
                nem_outputs.append(state)

            out_data = c_net(nem_outputs[-1][0],gamma)
            data_path = '/mnt/cloud_disk/yw/retrieval_test_vggm_shapenetcore/' + data[2][0].split('/', 5)[5]
            #data_path = '/mnt/cloud_disk/yw/gamma2/'+ data[2][0].split('/',5)[5][0:-3] + 'npz'
            np.save(data_path,out_data.cpu().detach().numpy())
            #np.savez(data_path, save_gamma=gamma.cpu().detach().numpy(), save_sort=out_data.cpu().detach().numpy())

    def train_kmean_threeview(self, n_epochs):

        best_acc = 0
        i_acc = 0
        self.model.train()
        for epoch in range(n_epochs):


            # plot learning rate
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.writer.add_scalar('params/lr', lr, epoch)

            # train one epoch
            out_data = None
            in_data = None
            for i, data in enumerate(self.train_loader):

                N, V, C, H, W = data[1].size()
                input_path = data[2]
                kmean_center_feature = np.zeros([N,3,C,H,W])
                for i in range(N):
                    path = input_path[i]
                    in_feature = data[1][i]

                    item_name = path.split('/')[-1].split('.')[0]
                    path_parient = path[:-len(item_name+'.npy')-len('features/')]
                    kmean_path = os.path.join(path_parient,'cluster_fea',item_name+'.npz')
                    kmean_fea = np.load(kmean_path)
                    feature_center = kmean_fea['center_feas']
                    feature_center = feature_center.reshape(-1,512,7,7)
                    kmean_center_feature[i]=feature_center
                in_data = kmean_center_feature
                in_data = Variable(torch.from_numpy(in_data)).cuda().float()


                target = Variable(data[0]).cuda().long()

                self.optimizer.zero_grad()

                out_data = self.model(in_data)

                loss = self.loss_fn(out_data, target)
                i_acc += 1
                self.writer.add_scalar('train/train_loss', loss, i_acc)

                pred = torch.max(out_data, 1)[1]
                results = pred == target
                correct_points = torch.sum(results.long())

                acc = correct_points.float() / results.size()[0]
                self.writer.add_scalar('train/train_overall_acc', acc, i_acc)

                loss.backward()
                self.optimizer.step()

                log_str = 'epoch %d, step %d: train_loss %.3f; train_acc %.3f' % (epoch + 1, i + 1, loss, acc)
                if (i + 1) % 1 == 0:
                    print(log_str)


            # evaluation
            if (epoch + 1) % 1 == 0:
                with torch.no_grad():
                    loss, val_overall_acc, val_mean_class_acc = self.update_validation_accuracy_kmean(epoch)
                self.writer.add_scalar('val/val_mean_class_acc', val_mean_class_acc, epoch + 1)
                self.writer.add_scalar('val/val_overall_acc', val_overall_acc, epoch + 1)
                self.writer.add_scalar('val/val_loss', loss, epoch + 1)

            # save best model
            if val_overall_acc > best_acc:
                best_acc = val_overall_acc
                self.model.save(self.log_dir, epoch)

            # adjust learning rate manually
            if epoch > 0 and (epoch + 1) % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5

        # export scalar data to JSON for external processing
        self.writer.export_scalars_to_json(self.log_dir + "/all_scalars.json")
        self.writer.close()


    def update_validation_accuracy_kmean(self, epoch):
        all_correct_points = 0
        all_points = 0

        # in_data = None
        # out_data = None
        # target = None

        wrong_class = np.zeros(40)
        samples_class = np.zeros(40)
        all_loss = 0

        self.model.eval()

        avgpool = nn.AvgPool1d(1, 1)

        total_time = 0.0
        total_print_time = 0.0
        all_target = []
        all_pred = []

        for _, data in enumerate(self.val_loader, 0):

            N, V, C, H, W = data[1].size()
            input_path = data[2]
            kmean_center_feature = np.zeros([N, 3, C, H, W])
            for i in range(N):
                path = input_path[i]
                in_feature = data[1][i]

                item_name = path.split('/')[-1].split('.')[0]
                path_parient = path[:-len(item_name + '.npy') - len('features/')]
                kmean_path = os.path.join(path_parient, 'cluster_fea', item_name + '.npz')
                kmean_fea = np.load(kmean_path)
                feature_center = kmean_fea['center_feas']
                feature_center = feature_center.reshape(-1, 512, 7, 7)
                kmean_center_feature[i] = feature_center
            in_data = kmean_center_feature
            in_data = Variable(torch.from_numpy(in_data)).cuda().float()
            target = Variable(data[0]).cuda()

            out_data = self.model(in_data)
            pred = torch.max(out_data, 1)[1]
            all_loss += self.loss_fn(out_data, target).cpu().data.numpy()
            results = pred == target

            for i in range(results.size()[0]):
                if not bool(results[i].cpu().data.numpy()):
                    wrong_class[target.cpu().data.numpy().astype('int')[i]] += 1
                samples_class[target.cpu().data.numpy().astype('int')[i]] += 1
            correct_points = torch.sum(results.long())

            all_correct_points += correct_points
            all_points += results.size()[0]

        print ('Total # of test models: ', all_points)
        val_mean_class_acc = np.mean((samples_class-wrong_class)/samples_class)
        acc = float(all_correct_points) / all_points
        val_overall_acc = acc.cpu().data.numpy()
        loss = all_loss / len(self.val_loader)

        print ('val mean class acc. : ', val_mean_class_acc)
        print ('val overall acc. : ', val_overall_acc)
        print ('val loss : ', loss)

        self.model.train()

        return loss, val_overall_acc, val_mean_class_acc

    def train_attn_fc_sort(self, n_epochs):

        best_acc = 0
        i_acc = 0
        self.model.train()
        for epoch in range(n_epochs):
            # permute data for mvcnn
            rand_idx = np.random.permutation(int(len(self.train_loader.dataset.filepaths) / self.num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.train_loader.dataset.filepaths[
                                     rand_idx[i] * self.num_views:(rand_idx[i] + 1) * self.num_views])
            self.train_loader.dataset.filepaths = filepaths_new

            # plot learning rate
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.writer.add_scalar('params/lr', lr, epoch)

            # train one epoch
            out_data = None
            in_data = None
            for i, data in enumerate(self.train_loader):

                if self.model_name == 'mvcnn':
                    N, V, C, H, W = data[1].size()
                    in_data = Variable(data[1]).view(-1, C, H, W).cuda()
                else:
                    in_data = Variable(data[1].cuda())
                target = Variable(data[0]).cuda().long()

                self.optimizer.zero_grad()


                out_data, out_data2 = self.model(in_data)

                loss = self.loss_fn(out_data, target)+ self.loss_fn(out_data2,target)

                self.writer.add_scalar('train/train_loss', loss, i_acc + i + 1)

                pred = torch.max(out_data2, 1)[1]
                results = pred == target
                correct_points = torch.sum(results.long())

                acc = correct_points.float() / results.size()[0]
                self.writer.add_scalar('train/train_overall_acc', acc, i_acc + i + 1)

                loss.backward()
                self.optimizer.step()

                log_str = 'epoch %d, step %d: train_loss %.3f; train_acc %.3f' % (epoch + 1, i + 1, loss, acc)
                if (i + 1) % 1 == 0:
                    print(log_str)
            i_acc += i

            # evaluation
            if (epoch + 1) % 1 == 0:
                with torch.no_grad():
                    loss, val_overall_acc, val_mean_class_acc = self.update_validation_accuracy_sort(epoch)
                self.writer.add_scalar('val/val_mean_class_acc', val_mean_class_acc, epoch + 1)
                self.writer.add_scalar('val/val_overall_acc', val_overall_acc, epoch + 1)
                self.writer.add_scalar('val/val_loss', loss, epoch + 1)

            # save best model
            if val_overall_acc > best_acc:
                best_acc = val_overall_acc
                self.model.save(self.log_dir, epoch)

            # adjust learning rate manually
            if epoch > 0 and (epoch + 1) % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5

        # export scalar data to JSON for external processing
        self.writer.export_scalars_to_json(self.log_dir + "/all_scalars.json")
        self.writer.close()

    def train_fourview(self, n_epochs):

        best_acc = 0
        i_acc = 0
        self.model.train()
        for epoch in range(n_epochs):
            # permute data for mvcnn
            rand_idx = np.random.permutation(int(len(self.train_loader.dataset.filepaths) / self.num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.train_loader.dataset.filepaths[
                                     rand_idx[i] * self.num_views:(rand_idx[i] + 1) * self.num_views])
            self.train_loader.dataset.filepaths = filepaths_new

            # plot learning rate
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.writer.add_scalar('params/lr', lr, epoch)

            # train one epoch
            out_data = None
            in_data = None
            for i, data in enumerate(self.train_loader):

                if self.model_name == 'mvcnn':
                    N, V, C, H, W = data[1].size()
                    data[1] = data[1][:,0:4,:,:,:].contiguous()
                    in_data = Variable(data[1]).view(-1, C, H, W).cuda()
                else:
                    in_data = Variable(data[1].cuda())

                target = Variable(data[0]).cuda().long()

                self.optimizer.zero_grad()

                out_data = self.model(in_data)

                loss = self.loss_fn(out_data, target)

                self.writer.add_scalar('train/train_loss', loss, i_acc + i + 1)

                pred = torch.max(out_data, 1)[1]
                results = pred == target
                correct_points = torch.sum(results.long())

                acc = correct_points.float() / results.size()[0]
                self.writer.add_scalar('train/train_overall_acc', acc, i_acc + i + 1)

                loss.backward()
                self.optimizer.step()

                log_str = 'epoch %d, step %d: train_loss %.3f; train_acc %.3f' % (epoch + 1, i + 1, loss, acc)
                if (i + 1) % 1 == 0:
                    print(log_str)
            i_acc += i

            # evaluation
            if (epoch + 1) % 1 == 0:
                with torch.no_grad():
                    loss, val_overall_acc, val_mean_class_acc = self.update_validation_accuracy_fourview(epoch)
                self.writer.add_scalar('val/val_mean_class_acc', val_mean_class_acc, epoch + 1)
                self.writer.add_scalar('val/val_overall_acc', val_overall_acc, epoch + 1)
                self.writer.add_scalar('val/val_loss', loss, epoch + 1)

            # save best model
            if val_overall_acc > best_acc:
                best_acc = val_overall_acc
                self.model.save(self.log_dir, epoch)

            # adjust learning rate manually
            if epoch > 0 and (epoch + 1) % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5

        # export scalar data to JSON for external processing
        self.writer.export_scalars_to_json(self.log_dir + "/all_scalars.json")
        self.writer.close()

    def update_validation_accuracy(self, epoch):
        all_correct_points = 0
        all_points = 0

        # in_data = None
        # out_data = None
        # target = None

        wrong_class = np.zeros(40)
        samples_class = np.zeros(40)
        all_loss = 0

        self.model.eval()

        avgpool = nn.AvgPool1d(1, 1)

        total_time = 0.0
        total_print_time = 0.0
        all_target = []
        all_pred = []

        for _, data in enumerate(self.val_loader, 0):

            if self.model_name == 'mvcnn':
                N,V,C,H,W = data[1].size()
                in_data = Variable(data[1]).view(-1,C,H,W).cuda()
            else:#'svcnn'
                in_data = Variable(data[1]).cuda()
            target = Variable(data[0]).cuda()

            out_data = self.model(in_data)
            pred = torch.max(out_data, 1)[1]
            all_loss += self.loss_fn(out_data, target).cpu().data.numpy()
            results = pred == target

            for i in range(results.size()[0]):
                if not bool(results[i].cpu().data.numpy()):
                    wrong_class[target.cpu().data.numpy().astype('int')[i]] += 1
                samples_class[target.cpu().data.numpy().astype('int')[i]] += 1
            correct_points = torch.sum(results.long())

            all_correct_points += correct_points
            all_points += results.size()[0]

        print ('Total # of test models: ', all_points)
        val_mean_class_acc = np.mean((samples_class-wrong_class)/samples_class)
        acc = all_correct_points.float() / all_points
        val_overall_acc = acc.cpu().data.numpy()
        loss = all_loss / len(self.val_loader)

        print ('val mean class acc. : ', val_mean_class_acc)
        print ('val overall acc. : ', val_overall_acc)
        print ('val loss : ', loss)

        self.model.train()

        return loss, val_overall_acc, val_mean_class_acc

    def update_validation_accuracy_sort(self, epoch):
        all_correct_points = 0
        all_points = 0

        # in_data = None
        # out_data = None
        # target = None

        wrong_class = np.zeros(40)
        samples_class = np.zeros(40)
        all_loss = 0

        self.model.eval()

        avgpool = nn.AvgPool1d(1, 1)

        total_time = 0.0
        total_print_time = 0.0
        all_target = []
        all_pred = []

        for _, data in enumerate(self.val_loader, 0):

            if self.model_name == 'mvcnn':
                N, V, C, H, W = data[1].size()
                in_data = Variable(data[1]).view(-1, C, H, W).cuda()
            else:  # 'svcnn'
                in_data = Variable(data[1]).cuda()
            target = Variable(data[0]).cuda()

            out_data2, out_data = self.model(in_data)
            pred = torch.max(out_data, 1)[1]
            all_loss += self.loss_fn(out_data, target).cpu().data.numpy()+self.loss_fn(out_data2,target).cpu().data.numpy()
            results = pred == target

            for i in range(results.size()[0]):
                if not bool(results[i].cpu().data.numpy()):
                    wrong_class[target.cpu().data.numpy().astype('int')[i]] += 1
                samples_class[target.cpu().data.numpy().astype('int')[i]] += 1
            correct_points = torch.sum(results.long())

            all_correct_points += correct_points
            all_points += results.size()[0]

        print('Total # of test models: ', all_points)
        val_mean_class_acc = np.mean((samples_class - wrong_class) / samples_class)
        acc = all_correct_points.float() / all_points
        val_overall_acc = acc.cpu().data.numpy()
        loss = all_loss / len(self.val_loader)

        print('val mean class acc. : ', val_mean_class_acc)
        print('val overall acc. : ', val_overall_acc)
        print('val loss : ', loss)

        self.model.train()

        return loss, val_overall_acc, val_mean_class_acc

    def update_validation_accuracy_fourview(self, epoch):
        all_correct_points = 0
        all_points = 0

        # in_data = None
        # out_data = None
        # target = None

        wrong_class = np.zeros(40)
        samples_class = np.zeros(40)
        all_loss = 0

        self.model.eval()

        avgpool = nn.AvgPool1d(1, 1)

        total_time = 0.0
        total_print_time = 0.0
        all_target = []
        all_pred = []

        for _, data in enumerate(self.val_loader, 0):

            if self.model_name == 'mvcnn':
                N,V,C,H,W = data[1].size()
                data[1] = data[1][:,0:4,:,:,:].contiguous()
                in_data = Variable(data[1]).view(-1,C,H,W).cuda()
            else:#'svcnn'
                in_data = Variable(data[1]).cuda()
            target = Variable(data[0]).cuda()

            out_data = self.model(in_data)
            pred = torch.max(out_data, 1)[1]
            all_loss += self.loss_fn(out_data, target).cpu().data.numpy()
            results = pred == target

            for i in range(results.size()[0]):
                if not bool(results[i].cpu().data.numpy()):
                    wrong_class[target.cpu().data.numpy().astype('int')[i]] += 1
                samples_class[target.cpu().data.numpy().astype('int')[i]] += 1
            correct_points = torch.sum(results.long())

            all_correct_points += correct_points
            all_points += results.size()[0]

        print ('Total # of test models: ', all_points)
        val_mean_class_acc = np.mean((samples_class-wrong_class)/samples_class)
        acc = all_correct_points.float() / all_points
        val_overall_acc = acc.cpu().data.numpy()
        loss = all_loss / len(self.val_loader)

        print ('val mean class acc. : ', val_mean_class_acc)
        print ('val overall acc. : ', val_overall_acc)
        print ('val loss : ', loss)

        self.model.train()

        return loss, val_overall_acc, val_mean_class_acc

