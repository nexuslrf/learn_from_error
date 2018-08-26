function [negtrain,postrain] = parttrain(negtrain,postrain,t,params,mode,batchSize,res,labels)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    negnum = 0 ;
    posnum = 0 ;
    tmpneg=[];
    tmppos=[];
    tmpnegtrain=[];
    tmppostrain=[];
        for b=1:batchSize
            maxprediction=find(res(end-1).x(:,:,:,b)==max(res(end-1).x(:,:,:,b)));
            maxlabel=find(labels(:,:,:,b)==max(labels(:,:,:,b)));
            if(maxprediction~=maxlabel)
                negnum = negnum +1 ;
                tmpneg(negnum)=t+b-1;
            else
                posnum = posnum + 1 ;
                tmppos(posnum)=t+b-1;
            end
        end  
    for errim=1:numel(tmpneg)
        tmpnegtrain(errim)=params.train(tmpneg(errim));
    end
    for rigim=1:numel(tmppos)
        tmppostrain(rigim)=params.train(tmppos(rigim));
    end
    negtrain=cat(2,negtrain,tmpnegtrain);
    postrain=cat(2,postrain,tmppostrain);
    clear tmpneg tmppos  tmpnegtrain tmppostrain;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

