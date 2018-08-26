function [negval,posval] = partval(negval,posval,t,params,mode,batchSize,res,labels)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    negnum = 0 ;
    posnum = 0 ;
    tmpneg=[];
    tmppos=[];
    tmpnegval=[];
    tmpposval=[];
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
        tmpnegval(errim)=params.val(tmpneg(errim));
    end
    for rigim=1:numel(tmppos)
        tmpposval(rigim)=params.val(tmppos(rigim));
    end
    negval=cat(2,negval,tmpnegval);
    posval=cat(2,posval,tmpposval);
    clear tmpneg tmppos  tmpnegval tmpposval;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

