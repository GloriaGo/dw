
module DimmWitted

_libpath = ""
_data_type = ""
_model_type = ""

_nogc = {0.0,0.0}

MR_PERCORE = 0
MR_PERNODE = 1
MR_PERMACHINE = 2
MR_SINGLETHREAD_DEBUG = 3

DR_FULL = 0
DR_SHARDING = 1

AC_ROW = 0
AC_COL = 1
AC_C2R = 2

function set_libpath(path)
	global _libpath
	_libpath = path
end   

function get_libpath()
	global _libpath
	return _libpath
end

function open{DATATYPE, MODELTYPE}(examples::Array{DATATYPE,2}, model::Array{MODELTYPE,1}, modelrepl, datarepl, acmethod)

	global _libpath, _dw, _data_type, _model_type, _nogc

	_examples = examples
	_model = model
	_examples_c = examples.'

	append!(_nogc, {_examples, _model, _examples_c})

	nrows = size(examples, 1)
	ncols = size(examples, 2)
	nmodelel = size(model, 1)

	_data_type = DATATYPE
	_model_type = MODELTYPE

	_dw = @eval ccall( ($(string("DenseDimmWitted_Open2")), $(_libpath)), Ptr{Void}, (Any, Any, Clonglong, Clonglong, Clonglong, Ptr{Void}, Ptr{Void}, Cint, Cint, Cint), $(Array{DATATYPE}), $(Array{MODELTYPE}), $(nrows), $(ncols), $(nmodelel), $(_examples_c), $(_model), $(modelrepl), $(datarepl), $(acmethod))

	println("[JULIA-DW] Created DimmWitted Object: ", _dw)

	return _dw
end

function check_is_safe(func, ret, parameter)

	const stdout = STDOUT
	const rd, wr = redirect_stdout()
	code_llvm(func, parameter)
	str = readavailable(rd)
	close(rd)
	redirect_stdout(STDERR)

	if contains(str, "alloc") || contains(replace(str, string("julia_",func), ""), "julia_")
		return false
	else
		return true
	end
end


function register_row(_dw, func, supress=false)

	is_safe = check_is_safe(func, Cdouble, (Array{_data_type,1}, Array{_model_type,1}))
	if is_safe == false && supress==false
		error("Your function contains LLVM LR `alloc` or `call` other julia functions. We cannot register this function because it protentially is not thread-safe. Use register_row(_dw",",",func,",true) to register this function AT YOUR OWN RISK!")
	end

	global _data_type, _model_type, _libpath, _nogc

	const func_c = cfunction(func, Cdouble, (Array{_data_type,1}, Array{_model_type,1}))

	append!(_nogc, {func_c, func})

	handle = @eval ccall(($(string("DenseDimmWitted_Register_Row2")), $(_libpath)), Cuint, (Ptr{Void}, Ptr{Void}), $(_dw), $(func_c)) 

	println("[JULIA-DW] Registered Row Function ", func, " Handle=", handle)

	return handle
end

function exec(_dw, func_handle)
	global _libpath

	rs = @eval ccall(($(string("DenseDimmWitted_Exec2")), $(_libpath)), Cdouble, (Ptr{Void}, Cuint), $(_dw), $(func_handle))

	return rs
end


export libpath, get_libpath, open, register_row, exec

end


