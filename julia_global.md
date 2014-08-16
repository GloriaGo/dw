---
layout: page
title: 
permalink: /julia_global/
---

## Julia Support: Can my gradient function accesses some global variables, e.g., stepsize?

In the example we just show in logistic regression [here](/dw/julia/), we note
that the gradient function uses the same stepsize (i.e., 0.00001) in all
iterations. In real applications, we might want to use different
stepsizes for different iterations; and more generally, we might also want
to use some global variables inside the gradient function.
In this tutorial, we will show you how to do this. The code can be found
[here](https://github.com/zhangce/dw/blob/master/examples/julia_lr_shareddata.jl).

**Pre-requisites...** To understand this tutorial, we assume that you have
already familiar with the [Julia walkthorugh](/dw/julia/), and knows how to
write a logistic regression model where the loss function and gradient function
does not have access to global variables.

### What we cannot do?

As of now, you **cannot** use the following ways that
seems natural.

{% highlight julia linenos%}
stepsize = 0.00001
function grad(row::Array{Cdouble,1}, model::Array{Cdouble,1})
  global stepsize
end
{% endhighlight %}

The reason is mentioned in [Julia's documentation](http://julia.readthedocs.org/en/latest/manual/calling-c-and-fortran-code/#accessing-global-variables). Therefore, in DimmWitted,
we provide the following workaround.

### The Current Workaround

As a workaround of accessing global variables, we extend the schema
of the loss and gradient function to also take as input one more
variable, which is an array of arbitrary immutable type.
For example, to pass the stepsize, we can define

{% highlight julia linenos%}
immutable SHARED_DATA
	stepsize::Cdouble
	decay::Cdouble
end
shared_data = Array(SHARED_DATA, 1)
shared_data[1] = SHARED_DATA(0.00001, 0.99)
{% endhighlight %}

Accordingly, the gradient function becomes

{% highlight julia linenos%}
function grad(row::Array, model::Array, _shared_data::Array{SHARED_DATA,1})
	const stepsize = _shared_data[1].stepsize
	const label = row[length(row)]
	const nfeat = length(model)
	d = 0.0
	for i = 1:nfeat
		d = d + row[i]*model[i]
	end
	d = exp(-d)
	Z = stepsize * (-label + 1.0/(1.0+d))
  	for i = 1:nfeat
  		model[i] = model[i] - row[i] * Z
  	end
	return 1.0
end
{% endhighlight %}

Note that, this gradient function takes as input 
a variable called `_shared_data`.

When creating the DimmWitted object, you need to pass
in the `shared_data` object that we just created as
the last argument

{% highlight julia linenos%}
dw = DimmWitted.open(examples, model, 
                DimmWitted.MR_PERMACHINE,    
                DimmWitted.DR_SHARDING,      
                DimmWitted.AC_ROW, shared_data)
{% endhighlight %}


The last twist you need to do is that you cannot
use `register_row` to register the function now,
instead, you need to use a function called
`register_row2`:

{% highlight julia linenos%}
handle_grad = DimmWitted.register_row2(dw, grad)
{% endhighlight %}
















