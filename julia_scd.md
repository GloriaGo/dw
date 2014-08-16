---
layout: page
title: 
permalink: /julia_scd/
---

## Julia Support: How to write other access methods in Julia for DimmWitted?

You can write in Julia all three access methods, i.e., row-wise, column-wise,
and column-to-row, that DimmWitted supported. In this tutorial,
we will show how to write a logistic regression model using SCD instead of SGD.
SCD for logistic regression is a column-to-row access method.
The code can be found
[here](https://github.com/zhangce/dw/blob/master/examples/julia_lr_scd.jl).

**Pre-requisites...** To understand this tutorial, we assume that you have
already familiar with the [Julia walkthorugh](/dw/julia/), and knows how to
write a logistic regression model with SGD.

### Revising Gradient Function

To change access methods, you do not need to chang the data, and therefore
you can use the same synthetic data set that we created for SGD. However,
you need to change the gradient function with a different signature:

{% highlight julia linenos%}
function grad(col::Array{Cdouble,1}, 
              _colid::Cint, 
              rows::Array{Array{Cdouble, 1}}, 
              model::Array{Cdouble,1})
{% endhighlight %}

Different from the row-wise gradient function, the column-to-row gradient
function takes as input four parameters, where `col`
is the array of one column, `_colid` is the index of
this column (start from 0), `rows` is a array of
rows that has non-zero element for column id=`colid`,
and `model` is the model. Given this signature,
we can write the gradient function as

{% highlight julia linenos%}
function grad(col, _colid, rows, model)
	colid = _colid + 1
	nfeat = length(model)
	nrows = length(rows)
	if colid > nfeat
		return 1.0
	end

	sum_term = 0.0
	pat_term = 0.0
	for ir = 1:length(rows)
		label = rows[ir][nfeat+1]
		d = 0.0
		for i = 1:nfeat
			d = d + rows[ir][i]*model[i]
		end
		sum_term = sum_term + label*rows[ir][colid]
		pat_term = pat_term + rows[ir][colid]*1.0/(1.0+exp(-d))
	end

	model[colid] = model[colid] - 0.00001* (-sum_term + pat_term)
	
	return 1.0
end
{% endhighlight %}

This function contains multiple components:

  - Line 2: Note that the variable `_colid` starts from 0, however, the index
  of Julia starts from 1, therefore, we create the variable `colid` to start
  from 1.
  - Line 3-4: Get the number of features and number of rows. 
  - Line 5-7: If `colid` is the last column (i.e., the label column), we do nothing.
  - Line 9-19: Calculate the gradient of the `colid`'th element in the model.
  - Line 21: Update the `colid`'th element in the model.

### Register a Column-to-row Function

The last twist that we need to do is when we register the function. Instead of
using `register_row`, we should use `register_c2r` as

{% highlight julia linenos%}
handle_grad = DimmWitted.register_c2r(dw, grad)
{% endhighlight %}

Also, when creating the DimmWitted object, we should use `DimmWitted.AC_C2R`
instead of `DimmWitted.AC_ROW`:

{% highlight julia linenos%}
dw = DimmWitted.open(examples, model, 
                DimmWitted.MR_PERMACHINE,    
                DimmWitted.DR_SHARDING,      
                DimmWitted.AC_C2R)
{% endhighlight %}

Note that for column-to-row access, you can register either row access function
or column-to-row function. For row (resp. column) access, you can only register
row (resp. column) function.










