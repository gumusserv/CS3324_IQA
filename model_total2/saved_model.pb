иц	
џЮ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
,
Exp
x"T
y"T"
Ttype:

2
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
С
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.11.02v2.11.0-rc2-15-g6290819256d8у
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
r
Adam/v/VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameAdam/v/Variable
k
#Adam/v/Variable/Read/ReadVariableOpReadVariableOpAdam/v/Variable*
_output_shapes
: *
dtype0
r
Adam/m/VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameAdam/m/Variable
k
#Adam/m/Variable/Read/ReadVariableOpReadVariableOpAdam/m/Variable*
_output_shapes
: *
dtype0
v
Adam/v/Variable_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/v/Variable_1
o
%Adam/v/Variable_1/Read/ReadVariableOpReadVariableOpAdam/v/Variable_1*
_output_shapes
: *
dtype0
v
Adam/m/Variable_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/m/Variable_1
o
%Adam/m/Variable_1/Read/ReadVariableOpReadVariableOpAdam/m/Variable_1*
_output_shapes
: *
dtype0
v
Adam/v/Variable_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/v/Variable_2
o
%Adam/v/Variable_2/Read/ReadVariableOpReadVariableOpAdam/v/Variable_2*
_output_shapes
: *
dtype0
v
Adam/m/Variable_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/m/Variable_2
o
%Adam/m/Variable_2/Read/ReadVariableOpReadVariableOpAdam/m/Variable_2*
_output_shapes
: *
dtype0
v
Adam/v/Variable_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/v/Variable_3
o
%Adam/v/Variable_3/Read/ReadVariableOpReadVariableOpAdam/v/Variable_3*
_output_shapes
: *
dtype0
v
Adam/m/Variable_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/m/Variable_3
o
%Adam/m/Variable_3/Read/ReadVariableOpReadVariableOpAdam/m/Variable_3*
_output_shapes
: *
dtype0
v
Adam/v/Variable_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/v/Variable_4
o
%Adam/v/Variable_4/Read/ReadVariableOpReadVariableOpAdam/v/Variable_4*
_output_shapes
: *
dtype0
v
Adam/m/Variable_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/m/Variable_4
o
%Adam/m/Variable_4/Read/ReadVariableOpReadVariableOpAdam/m/Variable_4*
_output_shapes
: *
dtype0

Adam/v/quality_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/v/quality_output/bias

.Adam/v/quality_output/bias/Read/ReadVariableOpReadVariableOpAdam/v/quality_output/bias*
_output_shapes
:*
dtype0

Adam/m/quality_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/m/quality_output/bias

.Adam/m/quality_output/bias/Read/ReadVariableOpReadVariableOpAdam/m/quality_output/bias*
_output_shapes
:*
dtype0

Adam/v/quality_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*-
shared_nameAdam/v/quality_output/kernel

0Adam/v/quality_output/kernel/Read/ReadVariableOpReadVariableOpAdam/v/quality_output/kernel*
_output_shapes

:@*
dtype0

Adam/m/quality_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*-
shared_nameAdam/m/quality_output/kernel

0Adam/m/quality_output/kernel/Read/ReadVariableOpReadVariableOpAdam/m/quality_output/kernel*
_output_shapes

:@*
dtype0
~
Adam/v/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/v/dense_5/bias
w
'Adam/v/dense_5/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_5/bias*
_output_shapes
:@*
dtype0
~
Adam/m/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/m/dense_5/bias
w
'Adam/m/dense_5/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_5/bias*
_output_shapes
:@*
dtype0

Adam/v/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*&
shared_nameAdam/v/dense_5/kernel

)Adam/v/dense_5/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_5/kernel*
_output_shapes
:	@*
dtype0

Adam/m/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*&
shared_nameAdam/m/dense_5/kernel

)Adam/m/dense_5/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_5/kernel*
_output_shapes
:	@*
dtype0

Adam/v/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/v/dense_4/bias
x
'Adam/v/dense_4/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_4/bias*
_output_shapes	
:*
dtype0

Adam/m/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/m/dense_4/bias
x
'Adam/m/dense_4/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_4/bias*
_output_shapes	
:*
dtype0

Adam/v/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
 *&
shared_nameAdam/v/dense_4/kernel

)Adam/v/dense_4/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_4/kernel* 
_output_shapes
:
 *
dtype0

Adam/m/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
 *&
shared_nameAdam/m/dense_4/kernel

)Adam/m/dense_4/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_4/kernel* 
_output_shapes
:
 *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
d
VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0
h

Variable_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_1
a
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
: *
dtype0
h

Variable_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_2
a
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes
: *
dtype0
h

Variable_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_3
a
Variable_3/Read/ReadVariableOpReadVariableOp
Variable_3*
_output_shapes
: *
dtype0
h

Variable_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_4
a
Variable_4/Read/ReadVariableOpReadVariableOp
Variable_4*
_output_shapes
: *
dtype0
~
quality_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namequality_output/bias
w
'quality_output/bias/Read/ReadVariableOpReadVariableOpquality_output/bias*
_output_shapes
:*
dtype0

quality_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_namequality_output/kernel

)quality_output/kernel/Read/ReadVariableOpReadVariableOpquality_output/kernel*
_output_shapes

:@*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:@*
dtype0
y
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*
shared_namedense_5/kernel
r
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes
:	@*
dtype0
q
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
j
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes	
:*
dtype0
z
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
 *
shared_namedense_4/kernel
s
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel* 
_output_shapes
:
 *
dtype0
|
serving_default_input_5Placeholder*(
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
|
serving_default_input_6Placeholder*(
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_5serving_default_input_6dense_4/kerneldense_4/biasdense_5/kerneldense_5/biasquality_output/kernelquality_output/bias
Variable_2
Variable_3
Variable_4
Variable_1Variable*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_38636

NoOpNoOp
I
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ЩH
valueПHBМH BЕH
Љ
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer-4
layer_with_weights-1
layer-5
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
І
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
 bias*
Ѕ
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
'_random_generator* 
І
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

.kernel
/bias*
Ѕ
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6_random_generator* 
І
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses

=kernel
>bias*
Ь
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses

Eparam1

Fparam2

Gparam3

Hparam4

Iparam5*
R
0
 1
.2
/3
=4
>5
E6
F7
G8
H9
I10*
R
0
 1
.2
/3
=4
>5
E6
F7
G8
H9
I10*
* 
А
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Otrace_0
Ptrace_1
Qtrace_2
Rtrace_3* 
6
Strace_0
Ttrace_1
Utrace_2
Vtrace_3* 
* 

W
_variables
X_iterations
Y_learning_rate
Z_index_dict
[
_momentums
\_velocities
]_update_step_xla*

^serving_default* 
* 
* 
* 

_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

dtrace_0* 

etrace_0* 

0
 1*

0
 1*
* 

fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

ktrace_0* 

ltrace_0* 
^X
VARIABLE_VALUEdense_4/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_4/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses* 

rtrace_0
strace_1* 

ttrace_0
utrace_1* 
* 

.0
/1*

.0
/1*
* 

vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses*

{trace_0* 

|trace_0* 
^X
VARIABLE_VALUEdense_5/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_5/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

}non_trainable_variables

~layers
metrics
 layer_regularization_losses
layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses* 

trace_0
trace_1* 

trace_0
trace_1* 
* 

=0
>1*

=0
>1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses*

trace_0* 

trace_0* 
e_
VARIABLE_VALUEquality_output/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEquality_output/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
'
E0
F1
G2
H3
I4*
'
E0
F1
G2
H3
I4*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses*

trace_0* 

trace_0* 
ZT
VARIABLE_VALUE
Variable_46layer_with_weights-3/param1/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUE
Variable_36layer_with_weights-3/param2/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUE
Variable_26layer_with_weights-3/param3/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUE
Variable_16layer_with_weights-3/param4/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEVariable6layer_with_weights-3/param5/.ATTRIBUTES/VARIABLE_VALUE*
* 
C
0
1
2
3
4
5
6
7
	8*

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
Ш
X0
1
2
3
4
5
6
7
8
9
10
11
 12
Ё13
Ђ14
Ѓ15
Є16
Ѕ17
І18
Ї19
Ј20
Љ21
Њ22*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
]
0
1
2
3
4
5
Ё6
Ѓ7
Ѕ8
Ї9
Љ10*
]
0
1
2
3
4
 5
Ђ6
Є7
І8
Ј9
Њ10*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
Ћ	variables
Ќ	keras_api

­total

Ўcount*
`Z
VARIABLE_VALUEAdam/m/dense_4/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_4/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/dense_4/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/dense_4/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_5/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_5/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/dense_5/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/dense_5/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEAdam/m/quality_output/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEAdam/v/quality_output/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/m/quality_output/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/v/quality_output/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/Variable_42optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/Variable_42optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/Variable_32optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/Variable_32optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/Variable_22optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/Variable_22optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/Variable_12optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/Variable_12optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/m/Variable2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/v/Variable2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*

­0
Ў1*

Ћ	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ы
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp)quality_output/kernel/Read/ReadVariableOp'quality_output/bias/Read/ReadVariableOpVariable_4/Read/ReadVariableOpVariable_3/Read/ReadVariableOpVariable_2/Read/ReadVariableOpVariable_1/Read/ReadVariableOpVariable/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp)Adam/m/dense_4/kernel/Read/ReadVariableOp)Adam/v/dense_4/kernel/Read/ReadVariableOp'Adam/m/dense_4/bias/Read/ReadVariableOp'Adam/v/dense_4/bias/Read/ReadVariableOp)Adam/m/dense_5/kernel/Read/ReadVariableOp)Adam/v/dense_5/kernel/Read/ReadVariableOp'Adam/m/dense_5/bias/Read/ReadVariableOp'Adam/v/dense_5/bias/Read/ReadVariableOp0Adam/m/quality_output/kernel/Read/ReadVariableOp0Adam/v/quality_output/kernel/Read/ReadVariableOp.Adam/m/quality_output/bias/Read/ReadVariableOp.Adam/v/quality_output/bias/Read/ReadVariableOp%Adam/m/Variable_4/Read/ReadVariableOp%Adam/v/Variable_4/Read/ReadVariableOp%Adam/m/Variable_3/Read/ReadVariableOp%Adam/v/Variable_3/Read/ReadVariableOp%Adam/m/Variable_2/Read/ReadVariableOp%Adam/v/Variable_2/Read/ReadVariableOp%Adam/m/Variable_1/Read/ReadVariableOp%Adam/v/Variable_1/Read/ReadVariableOp#Adam/m/Variable/Read/ReadVariableOp#Adam/v/Variable/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*2
Tin+
)2'	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_39113
т
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_4/kerneldense_4/biasdense_5/kerneldense_5/biasquality_output/kernelquality_output/bias
Variable_4
Variable_3
Variable_2
Variable_1Variable	iterationlearning_rateAdam/m/dense_4/kernelAdam/v/dense_4/kernelAdam/m/dense_4/biasAdam/v/dense_4/biasAdam/m/dense_5/kernelAdam/v/dense_5/kernelAdam/m/dense_5/biasAdam/v/dense_5/biasAdam/m/quality_output/kernelAdam/v/quality_output/kernelAdam/m/quality_output/biasAdam/v/quality_output/biasAdam/m/Variable_4Adam/v/Variable_4Adam/m/Variable_3Adam/v/Variable_3Adam/m/Variable_2Adam/v/Variable_2Adam/m/Variable_1Adam/v/Variable_1Adam/m/VariableAdam/v/Variabletotalcount*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_39234ск


c
D__inference_dropout_5_layer_call_and_return_conditional_losses_38365

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>І
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ѓ
b
)__inference_dropout_4_layer_call_fn_38853

inputs
identityЂStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_38398p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ЫJ
ч
B__inference_model_2_layer_call_and_return_conditional_losses_38810
inputs_0
inputs_1:
&dense_4_matmul_readvariableop_resource:
 6
'dense_4_biasadd_readvariableop_resource:	9
&dense_5_matmul_readvariableop_resource:	@5
'dense_5_biasadd_readvariableop_resource:@?
-quality_output_matmul_readvariableop_resource:@<
.quality_output_biasadd_readvariableop_resource:5
+mapping_layer_2_sub_readvariableop_resource: 1
'mapping_layer_2_readvariableop_resource: 3
)mapping_layer_2_readvariableop_1_resource: 3
)mapping_layer_2_readvariableop_2_resource: 7
-mapping_layer_2_add_2_readvariableop_resource: 
identityЂdense_4/BiasAdd/ReadVariableOpЂdense_4/MatMul/ReadVariableOpЂdense_5/BiasAdd/ReadVariableOpЂdense_5/MatMul/ReadVariableOpЂmapping_layer_2/ReadVariableOpЂ mapping_layer_2/ReadVariableOp_1Ђ mapping_layer_2/ReadVariableOp_2Ђ$mapping_layer_2/add_2/ReadVariableOpЂ"mapping_layer_2/sub/ReadVariableOpЂ%quality_output/BiasAdd/ReadVariableOpЂ$quality_output/MatMul/ReadVariableOp[
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatenate_2/concatConcatV2inputs_0inputs_1"concatenate_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:џџџџџџџџџ 
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0
dense_4/MatMulMatMulconcatenate_2/concat:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџg
dense_4/SigmoidSigmoiddense_4/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ\
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
dropout_4/dropout/MulMuldense_4/Sigmoid:y:0 dropout_4/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџZ
dropout_4/dropout/ShapeShapedense_4/Sigmoid:y:0*
T0*
_output_shapes
:Ё
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0e
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>Х
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ^
dropout_4/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    М
dropout_4/dropout/SelectV2SelectV2"dropout_4/dropout/GreaterEqual:z:0dropout_4/dropout/Mul:z:0"dropout_4/dropout/Const_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0
dense_5/MatMulMatMul#dropout_4/dropout/SelectV2:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
dense_5/SigmoidSigmoiddense_5/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@\
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
dropout_5/dropout/MulMuldense_5/Sigmoid:y:0 dropout_5/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@Z
dropout_5/dropout/ShapeShapedense_5/Sigmoid:y:0*
T0*
_output_shapes
: 
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@*
dtype0e
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>Ф
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@^
dropout_5/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Л
dropout_5/dropout/SelectV2SelectV2"dropout_5/dropout/GreaterEqual:z:0dropout_5/dropout/Mul:z:0"dropout_5/dropout/Const_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@
$quality_output/MatMul/ReadVariableOpReadVariableOp-quality_output_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Є
quality_output/MatMulMatMul#dropout_5/dropout/SelectV2:output:0,quality_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
%quality_output/BiasAdd/ReadVariableOpReadVariableOp.quality_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ѓ
quality_output/BiasAddBiasAddquality_output/MatMul:product:0-quality_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
"mapping_layer_2/sub/ReadVariableOpReadVariableOp+mapping_layer_2_sub_readvariableop_resource*
_output_shapes
: *
dtype0
mapping_layer_2/subSubquality_output/BiasAdd:output:0*mapping_layer_2/sub/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ~
mapping_layer_2/ReadVariableOpReadVariableOp'mapping_layer_2_readvariableop_resource*
_output_shapes
: *
dtype0
mapping_layer_2/mulMul&mapping_layer_2/ReadVariableOp:value:0mapping_layer_2/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџe
mapping_layer_2/ExpExpmapping_layer_2/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџZ
mapping_layer_2/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
mapping_layer_2/addAddV2mapping_layer_2/add/x:output:0mapping_layer_2/Exp:y:0*
T0*'
_output_shapes
:џџџџџџџџџ^
mapping_layer_2/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
mapping_layer_2/truedivRealDiv"mapping_layer_2/truediv/x:output:0mapping_layer_2/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ\
mapping_layer_2/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
mapping_layer_2/sub_1Sub mapping_layer_2/sub_1/x:output:0mapping_layer_2/truediv:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
 mapping_layer_2/ReadVariableOp_1ReadVariableOp)mapping_layer_2_readvariableop_1_resource*
_output_shapes
: *
dtype0
mapping_layer_2/mul_1Mul(mapping_layer_2/ReadVariableOp_1:value:0mapping_layer_2/sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
 mapping_layer_2/ReadVariableOp_2ReadVariableOp)mapping_layer_2_readvariableop_2_resource*
_output_shapes
: *
dtype0
mapping_layer_2/mul_2Mul(mapping_layer_2/ReadVariableOp_2:value:0quality_output/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
mapping_layer_2/add_1AddV2mapping_layer_2/mul_1:z:0mapping_layer_2/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
$mapping_layer_2/add_2/ReadVariableOpReadVariableOp-mapping_layer_2_add_2_readvariableop_resource*
_output_shapes
: *
dtype0
mapping_layer_2/add_2AddV2mapping_layer_2/add_1:z:0,mapping_layer_2/add_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџh
IdentityIdentitymapping_layer_2/add_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЪ
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^mapping_layer_2/ReadVariableOp!^mapping_layer_2/ReadVariableOp_1!^mapping_layer_2/ReadVariableOp_2%^mapping_layer_2/add_2/ReadVariableOp#^mapping_layer_2/sub/ReadVariableOp&^quality_output/BiasAdd/ReadVariableOp%^quality_output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
mapping_layer_2/ReadVariableOpmapping_layer_2/ReadVariableOp2D
 mapping_layer_2/ReadVariableOp_1 mapping_layer_2/ReadVariableOp_12D
 mapping_layer_2/ReadVariableOp_2 mapping_layer_2/ReadVariableOp_22L
$mapping_layer_2/add_2/ReadVariableOp$mapping_layer_2/add_2/ReadVariableOp2H
"mapping_layer_2/sub/ReadVariableOp"mapping_layer_2/sub/ReadVariableOp2N
%quality_output/BiasAdd/ReadVariableOp%quality_output/BiasAdd/ReadVariableOp2L
$quality_output/MatMul/ReadVariableOp$quality_output/MatMul/ReadVariableOp:R N
(
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_0:RN
(
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_1

Ъ
/__inference_mapping_layer_2_layer_call_fn_38951
x
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_mapping_layer_2_layer_call_and_return_conditional_losses_38281o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:џџџџџџџџџ

_user_specified_namex
%
П
B__inference_model_2_layer_call_and_return_conditional_losses_38604
input_5
input_6!
dense_4_38575:
 
dense_4_38577:	 
dense_5_38581:	@
dense_5_38583:@&
quality_output_38587:@"
quality_output_38589:
mapping_layer_2_38592: 
mapping_layer_2_38594: 
mapping_layer_2_38596: 
mapping_layer_2_38598: 
mapping_layer_2_38600: 
identityЂdense_4/StatefulPartitionedCallЂdense_5/StatefulPartitionedCallЂ!dropout_4/StatefulPartitionedCallЂ!dropout_5/StatefulPartitionedCallЂ'mapping_layer_2/StatefulPartitionedCallЂ&quality_output/StatefulPartitionedCallЭ
concatenate_2/PartitionedCallPartitionedCallinput_5input_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_38188
dense_4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_4_38575dense_4_38577*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_38201ь
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_38398
dense_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_5_38581dense_5_38583*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_38225
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_38365Љ
&quality_output/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0quality_output_38587quality_output_38589*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_quality_output_layer_call_and_return_conditional_losses_38248§
'mapping_layer_2/StatefulPartitionedCallStatefulPartitionedCall/quality_output/StatefulPartitionedCall:output:0mapping_layer_2_38592mapping_layer_2_38594mapping_layer_2_38596mapping_layer_2_38598mapping_layer_2_38600*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_mapping_layer_2_layer_call_and_return_conditional_losses_38281
IdentityIdentity0mapping_layer_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЅ
NoOpNoOp ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall(^mapping_layer_2/StatefulPartitionedCall'^quality_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2R
'mapping_layer_2/StatefulPartitionedCall'mapping_layer_2/StatefulPartitionedCall2P
&quality_output/StatefulPartitionedCall&quality_output/StatefulPartitionedCall:Q M
(
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_5:QM
(
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_6
Ё
E
)__inference_dropout_4_layer_call_fn_38848

inputs
identityА
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_38212a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
С

'__inference_dense_5_layer_call_fn_38879

inputs
unknown:	@
	unknown_0:@
identityЂStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_38225o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ь

.__inference_quality_output_layer_call_fn_38926

inputs
unknown:@
	unknown_0:
identityЂStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_quality_output_layer_call_and_return_conditional_losses_38248o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ч
t
H__inference_concatenate_2_layer_call_and_return_conditional_losses_38823
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :x
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:џџџџџџџџџ X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџ:џџџџџџџџџ:R N
(
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_0:RN
(
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_1
Ѓ

J__inference_mapping_layer_2_layer_call_and_return_conditional_losses_38978
x%
sub_readvariableop_resource: !
readvariableop_resource: #
readvariableop_1_resource: #
readvariableop_2_resource: '
add_2_readvariableop_resource: 
identityЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2Ђadd_2/ReadVariableOpЂsub/ReadVariableOpf
sub/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes
: *
dtype0[
subSubxsub/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0]
mulMulReadVariableOp:value:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџE
ExpExpmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?W
addAddV2add/x:output:0Exp:y:0*
T0*'
_output_shapes
:џџџџџџџџџN
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
truedivRealDivtruediv/x:output:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?]
sub_1Subsub_1/x:output:0truediv:z:0*
T0*'
_output_shapes
:џџџџџџџџџb
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0c
mul_1MulReadVariableOp_1:value:0	sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџb
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype0[
mul_2MulReadVariableOp_2:value:0x*
T0*'
_output_shapes
:џџџџџџџџџV
add_1AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџj
add_2/ReadVariableOpReadVariableOpadd_2_readvariableop_resource*
_output_shapes
: *
dtype0i
add_2AddV2	add_1:z:0add_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentity	add_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЉ
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^add_2/ReadVariableOp^sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22,
add_2/ReadVariableOpadd_2/ReadVariableOp2(
sub/ReadVariableOpsub/ReadVariableOp:J F
'
_output_shapes
:џџџџџџџџџ

_user_specified_namex
КH

__inference__traced_save_39113
file_prefix-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop4
0savev2_quality_output_kernel_read_readvariableop2
.savev2_quality_output_bias_read_readvariableop)
%savev2_variable_4_read_readvariableop)
%savev2_variable_3_read_readvariableop)
%savev2_variable_2_read_readvariableop)
%savev2_variable_1_read_readvariableop'
#savev2_variable_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableop4
0savev2_adam_m_dense_4_kernel_read_readvariableop4
0savev2_adam_v_dense_4_kernel_read_readvariableop2
.savev2_adam_m_dense_4_bias_read_readvariableop2
.savev2_adam_v_dense_4_bias_read_readvariableop4
0savev2_adam_m_dense_5_kernel_read_readvariableop4
0savev2_adam_v_dense_5_kernel_read_readvariableop2
.savev2_adam_m_dense_5_bias_read_readvariableop2
.savev2_adam_v_dense_5_bias_read_readvariableop;
7savev2_adam_m_quality_output_kernel_read_readvariableop;
7savev2_adam_v_quality_output_kernel_read_readvariableop9
5savev2_adam_m_quality_output_bias_read_readvariableop9
5savev2_adam_v_quality_output_bias_read_readvariableop0
,savev2_adam_m_variable_4_read_readvariableop0
,savev2_adam_v_variable_4_read_readvariableop0
,savev2_adam_m_variable_3_read_readvariableop0
,savev2_adam_v_variable_3_read_readvariableop0
,savev2_adam_m_variable_2_read_readvariableop0
,savev2_adam_v_variable_2_read_readvariableop0
,savev2_adam_m_variable_1_read_readvariableop0
,savev2_adam_v_variable_1_read_readvariableop.
*savev2_adam_m_variable_read_readvariableop.
*savev2_adam_v_variable_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ў
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*з
valueЭBЪ&B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/param1/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/param2/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/param3/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/param4/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/param5/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЙ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop0savev2_quality_output_kernel_read_readvariableop.savev2_quality_output_bias_read_readvariableop%savev2_variable_4_read_readvariableop%savev2_variable_3_read_readvariableop%savev2_variable_2_read_readvariableop%savev2_variable_1_read_readvariableop#savev2_variable_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop0savev2_adam_m_dense_4_kernel_read_readvariableop0savev2_adam_v_dense_4_kernel_read_readvariableop.savev2_adam_m_dense_4_bias_read_readvariableop.savev2_adam_v_dense_4_bias_read_readvariableop0savev2_adam_m_dense_5_kernel_read_readvariableop0savev2_adam_v_dense_5_kernel_read_readvariableop.savev2_adam_m_dense_5_bias_read_readvariableop.savev2_adam_v_dense_5_bias_read_readvariableop7savev2_adam_m_quality_output_kernel_read_readvariableop7savev2_adam_v_quality_output_kernel_read_readvariableop5savev2_adam_m_quality_output_bias_read_readvariableop5savev2_adam_v_quality_output_bias_read_readvariableop,savev2_adam_m_variable_4_read_readvariableop,savev2_adam_v_variable_4_read_readvariableop,savev2_adam_m_variable_3_read_readvariableop,savev2_adam_v_variable_3_read_readvariableop,savev2_adam_m_variable_2_read_readvariableop,savev2_adam_v_variable_2_read_readvariableop,savev2_adam_m_variable_1_read_readvariableop,savev2_adam_v_variable_1_read_readvariableop*savev2_adam_m_variable_read_readvariableop*savev2_adam_v_variable_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *4
dtypes*
(2&	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Г
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*л
_input_shapesЩ
Ц: :
 ::	@:@:@:: : : : : : : :
 :
 :::	@:	@:@:@:@:@::: : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
 :!

_output_shapes	
::%!

_output_shapes
:	@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
 :&"
 
_output_shapes
:
 :!

_output_shapes	
::!

_output_shapes	
::%!

_output_shapes
:	@:%!

_output_shapes
:	@: 

_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@:$ 

_output_shapes

:@: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: 
Ь	
њ
I__inference_quality_output_layer_call_and_return_conditional_losses_38936

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs


є
B__inference_dense_5_layer_call_and_return_conditional_losses_38890

inputs1
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ё
ћ
'__inference_model_2_layer_call_fn_38536
input_5
input_6
unknown:
 
	unknown_0:	
	unknown_1:	@
	unknown_2:@
	unknown_3:@
	unknown_4:
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 
identityЂStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinput_5input_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_38483o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_5:QM
(
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_6
"
ї
B__inference_model_2_layer_call_and_return_conditional_losses_38294

inputs
inputs_1!
dense_4_38202:
 
dense_4_38204:	 
dense_5_38226:	@
dense_5_38228:@&
quality_output_38249:@"
quality_output_38251:
mapping_layer_2_38282: 
mapping_layer_2_38284: 
mapping_layer_2_38286: 
mapping_layer_2_38288: 
mapping_layer_2_38290: 
identityЂdense_4/StatefulPartitionedCallЂdense_5/StatefulPartitionedCallЂ'mapping_layer_2/StatefulPartitionedCallЂ&quality_output/StatefulPartitionedCallЭ
concatenate_2/PartitionedCallPartitionedCallinputsinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_38188
dense_4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_4_38202dense_4_38204*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_38201м
dropout_4/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_38212
dense_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_5_38226dense_5_38228*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_38225л
dropout_5/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_38236Ё
&quality_output/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0quality_output_38249quality_output_38251*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_quality_output_layer_call_and_return_conditional_losses_38248§
'mapping_layer_2/StatefulPartitionedCallStatefulPartitionedCall/quality_output/StatefulPartitionedCall:output:0mapping_layer_2_38282mapping_layer_2_38284mapping_layer_2_38286mapping_layer_2_38288mapping_layer_2_38290*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_mapping_layer_2_layer_call_and_return_conditional_losses_38281
IdentityIdentity0mapping_layer_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџн
NoOpNoOp ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall(^mapping_layer_2/StatefulPartitionedCall'^quality_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2R
'mapping_layer_2/StatefulPartitionedCall'mapping_layer_2/StatefulPartitionedCall2P
&quality_output/StatefulPartitionedCall&quality_output/StatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:PL
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Є

і
B__inference_dense_4_layer_call_and_return_conditional_losses_38201

inputs2
matmul_readvariableop_resource:
 .
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџW
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
А
Y
-__inference_concatenate_2_layer_call_fn_38816
inputs_0
inputs_1
identityС
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_38188a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџ:џџџџџџџџџ:R N
(
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_0:RN
(
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_1
ћ

ї
#__inference_signature_wrapper_38636
input_5
input_6
unknown:
 
	unknown_0:	
	unknown_1:	@
	unknown_2:@
	unknown_3:@
	unknown_4:
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 
identityЂStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinput_5input_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_38172o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_5:QM
(
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_6
ѕ
д
!__inference__traced_restore_39234
file_prefix3
assignvariableop_dense_4_kernel:
 .
assignvariableop_1_dense_4_bias:	4
!assignvariableop_2_dense_5_kernel:	@-
assignvariableop_3_dense_5_bias:@:
(assignvariableop_4_quality_output_kernel:@4
&assignvariableop_5_quality_output_bias:'
assignvariableop_6_variable_4: '
assignvariableop_7_variable_3: '
assignvariableop_8_variable_2: '
assignvariableop_9_variable_1: &
assignvariableop_10_variable: '
assignvariableop_11_iteration:	 +
!assignvariableop_12_learning_rate: =
)assignvariableop_13_adam_m_dense_4_kernel:
 =
)assignvariableop_14_adam_v_dense_4_kernel:
 6
'assignvariableop_15_adam_m_dense_4_bias:	6
'assignvariableop_16_adam_v_dense_4_bias:	<
)assignvariableop_17_adam_m_dense_5_kernel:	@<
)assignvariableop_18_adam_v_dense_5_kernel:	@5
'assignvariableop_19_adam_m_dense_5_bias:@5
'assignvariableop_20_adam_v_dense_5_bias:@B
0assignvariableop_21_adam_m_quality_output_kernel:@B
0assignvariableop_22_adam_v_quality_output_kernel:@<
.assignvariableop_23_adam_m_quality_output_bias:<
.assignvariableop_24_adam_v_quality_output_bias:/
%assignvariableop_25_adam_m_variable_4: /
%assignvariableop_26_adam_v_variable_4: /
%assignvariableop_27_adam_m_variable_3: /
%assignvariableop_28_adam_v_variable_3: /
%assignvariableop_29_adam_m_variable_2: /
%assignvariableop_30_adam_v_variable_2: /
%assignvariableop_31_adam_m_variable_1: /
%assignvariableop_32_adam_v_variable_1: -
#assignvariableop_33_adam_m_variable: -
#assignvariableop_34_adam_v_variable: #
assignvariableop_35_total: #
assignvariableop_36_count: 
identity_38ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Б
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*з
valueЭBЪ&B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/param1/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/param2/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/param3/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/param4/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/param5/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHМ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B п
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ў
_output_shapes
::::::::::::::::::::::::::::::::::::::*4
dtypes*
(2&	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOpAssignVariableOpassignvariableop_dense_4_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Ж
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_4_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_5_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ж
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_5_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_4AssignVariableOp(assignvariableop_4_quality_output_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_5AssignVariableOp&assignvariableop_5_quality_output_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_6AssignVariableOpassignvariableop_6_variable_4Identity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_7AssignVariableOpassignvariableop_7_variable_3Identity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_8AssignVariableOpassignvariableop_8_variable_2Identity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_9AssignVariableOpassignvariableop_9_variable_1Identity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Е
AssignVariableOp_10AssignVariableOpassignvariableop_10_variableIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0	*
_output_shapes
:Ж
AssignVariableOp_11AssignVariableOpassignvariableop_11_iterationIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_12AssignVariableOp!assignvariableop_12_learning_rateIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_13AssignVariableOp)assignvariableop_13_adam_m_dense_4_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_14AssignVariableOp)assignvariableop_14_adam_v_dense_4_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_15AssignVariableOp'assignvariableop_15_adam_m_dense_4_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_16AssignVariableOp'assignvariableop_16_adam_v_dense_4_biasIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_m_dense_5_kernelIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_18AssignVariableOp)assignvariableop_18_adam_v_dense_5_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_19AssignVariableOp'assignvariableop_19_adam_m_dense_5_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_v_dense_5_biasIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_21AssignVariableOp0assignvariableop_21_adam_m_quality_output_kernelIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_22AssignVariableOp0assignvariableop_22_adam_v_quality_output_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_23AssignVariableOp.assignvariableop_23_adam_m_quality_output_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_24AssignVariableOp.assignvariableop_24_adam_v_quality_output_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_25AssignVariableOp%assignvariableop_25_adam_m_variable_4Identity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_26AssignVariableOp%assignvariableop_26_adam_v_variable_4Identity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_27AssignVariableOp%assignvariableop_27_adam_m_variable_3Identity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_28AssignVariableOp%assignvariableop_28_adam_v_variable_3Identity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_29AssignVariableOp%assignvariableop_29_adam_m_variable_2Identity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_30AssignVariableOp%assignvariableop_30_adam_v_variable_2Identity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_31AssignVariableOp%assignvariableop_31_adam_m_variable_1Identity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_32AssignVariableOp%assignvariableop_32_adam_v_variable_1Identity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_33AssignVariableOp#assignvariableop_33_adam_m_variableIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_34AssignVariableOp#assignvariableop_34_adam_v_variableIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_35AssignVariableOpassignvariableop_35_totalIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_36AssignVariableOpassignvariableop_36_countIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 §
Identity_37Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_38IdentityIdentity_37:output:0^NoOp_1*
T0*
_output_shapes
: ъ
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_38Identity_38:output:0*_
_input_shapesN
L: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
л
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_38212

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

E
)__inference_dropout_5_layer_call_fn_38895

inputs
identityЏ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_38236`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs


c
D__inference_dropout_4_layer_call_and_return_conditional_losses_38398

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>Ї
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ї
§
'__inference_model_2_layer_call_fn_38692
inputs_0
inputs_1
unknown:
 
	unknown_0:	
	unknown_1:	@
	unknown_2:@
	unknown_3:@
	unknown_4:
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 
identityЂStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_38483o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_0:RN
(
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_1
Ѓ

J__inference_mapping_layer_2_layer_call_and_return_conditional_losses_38281
x%
sub_readvariableop_resource: !
readvariableop_resource: #
readvariableop_1_resource: #
readvariableop_2_resource: '
add_2_readvariableop_resource: 
identityЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2Ђadd_2/ReadVariableOpЂsub/ReadVariableOpf
sub/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes
: *
dtype0[
subSubxsub/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0]
mulMulReadVariableOp:value:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџE
ExpExpmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?W
addAddV2add/x:output:0Exp:y:0*
T0*'
_output_shapes
:џџџџџџџџџN
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
truedivRealDivtruediv/x:output:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?]
sub_1Subsub_1/x:output:0truediv:z:0*
T0*'
_output_shapes
:џџџџџџџџџb
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0c
mul_1MulReadVariableOp_1:value:0	sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџb
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype0[
mul_2MulReadVariableOp_2:value:0x*
T0*'
_output_shapes
:џџџџџџџџџV
add_1AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџj
add_2/ReadVariableOpReadVariableOpadd_2_readvariableop_resource*
_output_shapes
: *
dtype0i
add_2AddV2	add_1:z:0add_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentity	add_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЉ
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^add_2/ReadVariableOp^sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22,
add_2/ReadVariableOpadd_2/ReadVariableOp2(
sub/ReadVariableOpsub/ReadVariableOp:J F
'
_output_shapes
:џџџџџџџџџ

_user_specified_namex


є
B__inference_dense_5_layer_call_and_return_conditional_losses_38225

inputs1
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ь	
њ
I__inference_quality_output_layer_call_and_return_conditional_losses_38248

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
"
ї
B__inference_model_2_layer_call_and_return_conditional_losses_38570
input_5
input_6!
dense_4_38541:
 
dense_4_38543:	 
dense_5_38547:	@
dense_5_38549:@&
quality_output_38553:@"
quality_output_38555:
mapping_layer_2_38558: 
mapping_layer_2_38560: 
mapping_layer_2_38562: 
mapping_layer_2_38564: 
mapping_layer_2_38566: 
identityЂdense_4/StatefulPartitionedCallЂdense_5/StatefulPartitionedCallЂ'mapping_layer_2/StatefulPartitionedCallЂ&quality_output/StatefulPartitionedCallЭ
concatenate_2/PartitionedCallPartitionedCallinput_5input_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_38188
dense_4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_4_38541dense_4_38543*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_38201м
dropout_4/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_38212
dense_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_5_38547dense_5_38549*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_38225л
dropout_5/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_38236Ё
&quality_output/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0quality_output_38553quality_output_38555*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_quality_output_layer_call_and_return_conditional_losses_38248§
'mapping_layer_2/StatefulPartitionedCallStatefulPartitionedCall/quality_output/StatefulPartitionedCall:output:0mapping_layer_2_38558mapping_layer_2_38560mapping_layer_2_38562mapping_layer_2_38564mapping_layer_2_38566*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_mapping_layer_2_layer_call_and_return_conditional_losses_38281
IdentityIdentity0mapping_layer_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџн
NoOpNoOp ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall(^mapping_layer_2/StatefulPartitionedCall'^quality_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2R
'mapping_layer_2/StatefulPartitionedCall'mapping_layer_2/StatefulPartitionedCall2P
&quality_output/StatefulPartitionedCall&quality_output/StatefulPartitionedCall:Q M
(
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_5:QM
(
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_6
Х

'__inference_dense_4_layer_call_fn_38832

inputs
unknown:
 
	unknown_0:	
identityЂStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_38201p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
з
b
D__inference_dropout_5_layer_call_and_return_conditional_losses_38905

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
з
b
D__inference_dropout_5_layer_call_and_return_conditional_losses_38236

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
л
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_38858

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
%
П
B__inference_model_2_layer_call_and_return_conditional_losses_38483

inputs
inputs_1!
dense_4_38454:
 
dense_4_38456:	 
dense_5_38460:	@
dense_5_38462:@&
quality_output_38466:@"
quality_output_38468:
mapping_layer_2_38471: 
mapping_layer_2_38473: 
mapping_layer_2_38475: 
mapping_layer_2_38477: 
mapping_layer_2_38479: 
identityЂdense_4/StatefulPartitionedCallЂdense_5/StatefulPartitionedCallЂ!dropout_4/StatefulPartitionedCallЂ!dropout_5/StatefulPartitionedCallЂ'mapping_layer_2/StatefulPartitionedCallЂ&quality_output/StatefulPartitionedCallЭ
concatenate_2/PartitionedCallPartitionedCallinputsinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_38188
dense_4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_4_38454dense_4_38456*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_38201ь
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_38398
dense_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_5_38460dense_5_38462*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_38225
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_38365Љ
&quality_output/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0quality_output_38466quality_output_38468*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_quality_output_layer_call_and_return_conditional_losses_38248§
'mapping_layer_2/StatefulPartitionedCallStatefulPartitionedCall/quality_output/StatefulPartitionedCall:output:0mapping_layer_2_38471mapping_layer_2_38473mapping_layer_2_38475mapping_layer_2_38477mapping_layer_2_38479*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_mapping_layer_2_layer_call_and_return_conditional_losses_38281
IdentityIdentity0mapping_layer_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЅ
NoOpNoOp ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall(^mapping_layer_2/StatefulPartitionedCall'^quality_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2R
'mapping_layer_2/StatefulPartitionedCall'mapping_layer_2/StatefulPartitionedCall2P
&quality_output/StatefulPartitionedCall&quality_output/StatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:PL
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
П
r
H__inference_concatenate_2_layer_call_and_return_conditional_losses_38188

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :v
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:џџџџџџџџџ X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџ:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:PL
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ё
ћ
'__inference_model_2_layer_call_fn_38319
input_5
input_6
unknown:
 
	unknown_0:	
	unknown_1:	@
	unknown_2:@
	unknown_3:@
	unknown_4:
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 
identityЂStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinput_5input_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_38294o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_5:QM
(
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_6


c
D__inference_dropout_4_layer_call_and_return_conditional_losses_38870

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>Ї
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Є;
ч
B__inference_model_2_layer_call_and_return_conditional_losses_38744
inputs_0
inputs_1:
&dense_4_matmul_readvariableop_resource:
 6
'dense_4_biasadd_readvariableop_resource:	9
&dense_5_matmul_readvariableop_resource:	@5
'dense_5_biasadd_readvariableop_resource:@?
-quality_output_matmul_readvariableop_resource:@<
.quality_output_biasadd_readvariableop_resource:5
+mapping_layer_2_sub_readvariableop_resource: 1
'mapping_layer_2_readvariableop_resource: 3
)mapping_layer_2_readvariableop_1_resource: 3
)mapping_layer_2_readvariableop_2_resource: 7
-mapping_layer_2_add_2_readvariableop_resource: 
identityЂdense_4/BiasAdd/ReadVariableOpЂdense_4/MatMul/ReadVariableOpЂdense_5/BiasAdd/ReadVariableOpЂdense_5/MatMul/ReadVariableOpЂmapping_layer_2/ReadVariableOpЂ mapping_layer_2/ReadVariableOp_1Ђ mapping_layer_2/ReadVariableOp_2Ђ$mapping_layer_2/add_2/ReadVariableOpЂ"mapping_layer_2/sub/ReadVariableOpЂ%quality_output/BiasAdd/ReadVariableOpЂ$quality_output/MatMul/ReadVariableOp[
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatenate_2/concatConcatV2inputs_0inputs_1"concatenate_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:џџџџџџџџџ 
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0
dense_4/MatMulMatMulconcatenate_2/concat:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџg
dense_4/SigmoidSigmoiddense_4/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџf
dropout_4/IdentityIdentitydense_4/Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџ
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0
dense_5/MatMulMatMuldropout_4/Identity:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
dense_5/SigmoidSigmoiddense_5/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@e
dropout_5/IdentityIdentitydense_5/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@
$quality_output/MatMul/ReadVariableOpReadVariableOp-quality_output_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
quality_output/MatMulMatMuldropout_5/Identity:output:0,quality_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
%quality_output/BiasAdd/ReadVariableOpReadVariableOp.quality_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ѓ
quality_output/BiasAddBiasAddquality_output/MatMul:product:0-quality_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
"mapping_layer_2/sub/ReadVariableOpReadVariableOp+mapping_layer_2_sub_readvariableop_resource*
_output_shapes
: *
dtype0
mapping_layer_2/subSubquality_output/BiasAdd:output:0*mapping_layer_2/sub/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ~
mapping_layer_2/ReadVariableOpReadVariableOp'mapping_layer_2_readvariableop_resource*
_output_shapes
: *
dtype0
mapping_layer_2/mulMul&mapping_layer_2/ReadVariableOp:value:0mapping_layer_2/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџe
mapping_layer_2/ExpExpmapping_layer_2/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџZ
mapping_layer_2/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
mapping_layer_2/addAddV2mapping_layer_2/add/x:output:0mapping_layer_2/Exp:y:0*
T0*'
_output_shapes
:џџџџџџџџџ^
mapping_layer_2/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
mapping_layer_2/truedivRealDiv"mapping_layer_2/truediv/x:output:0mapping_layer_2/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ\
mapping_layer_2/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
mapping_layer_2/sub_1Sub mapping_layer_2/sub_1/x:output:0mapping_layer_2/truediv:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
 mapping_layer_2/ReadVariableOp_1ReadVariableOp)mapping_layer_2_readvariableop_1_resource*
_output_shapes
: *
dtype0
mapping_layer_2/mul_1Mul(mapping_layer_2/ReadVariableOp_1:value:0mapping_layer_2/sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
 mapping_layer_2/ReadVariableOp_2ReadVariableOp)mapping_layer_2_readvariableop_2_resource*
_output_shapes
: *
dtype0
mapping_layer_2/mul_2Mul(mapping_layer_2/ReadVariableOp_2:value:0quality_output/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
mapping_layer_2/add_1AddV2mapping_layer_2/mul_1:z:0mapping_layer_2/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
$mapping_layer_2/add_2/ReadVariableOpReadVariableOp-mapping_layer_2_add_2_readvariableop_resource*
_output_shapes
: *
dtype0
mapping_layer_2/add_2AddV2mapping_layer_2/add_1:z:0,mapping_layer_2/add_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџh
IdentityIdentitymapping_layer_2/add_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЪ
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^mapping_layer_2/ReadVariableOp!^mapping_layer_2/ReadVariableOp_1!^mapping_layer_2/ReadVariableOp_2%^mapping_layer_2/add_2/ReadVariableOp#^mapping_layer_2/sub/ReadVariableOp&^quality_output/BiasAdd/ReadVariableOp%^quality_output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
mapping_layer_2/ReadVariableOpmapping_layer_2/ReadVariableOp2D
 mapping_layer_2/ReadVariableOp_1 mapping_layer_2/ReadVariableOp_12D
 mapping_layer_2/ReadVariableOp_2 mapping_layer_2/ReadVariableOp_22L
$mapping_layer_2/add_2/ReadVariableOp$mapping_layer_2/add_2/ReadVariableOp2H
"mapping_layer_2/sub/ReadVariableOp"mapping_layer_2/sub/ReadVariableOp2N
%quality_output/BiasAdd/ReadVariableOp%quality_output/BiasAdd/ReadVariableOp2L
$quality_output/MatMul/ReadVariableOp$quality_output/MatMul/ReadVariableOp:R N
(
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_0:RN
(
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_1
Ї
§
'__inference_model_2_layer_call_fn_38664
inputs_0
inputs_1
unknown:
 
	unknown_0:	
	unknown_1:	@
	unknown_2:@
	unknown_3:@
	unknown_4:
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 
identityЂStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_38294o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_0:RN
(
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_1
еC
ѓ	
 __inference__wrapped_model_38172
input_5
input_6B
.model_2_dense_4_matmul_readvariableop_resource:
 >
/model_2_dense_4_biasadd_readvariableop_resource:	A
.model_2_dense_5_matmul_readvariableop_resource:	@=
/model_2_dense_5_biasadd_readvariableop_resource:@G
5model_2_quality_output_matmul_readvariableop_resource:@D
6model_2_quality_output_biasadd_readvariableop_resource:=
3model_2_mapping_layer_2_sub_readvariableop_resource: 9
/model_2_mapping_layer_2_readvariableop_resource: ;
1model_2_mapping_layer_2_readvariableop_1_resource: ;
1model_2_mapping_layer_2_readvariableop_2_resource: ?
5model_2_mapping_layer_2_add_2_readvariableop_resource: 
identityЂ&model_2/dense_4/BiasAdd/ReadVariableOpЂ%model_2/dense_4/MatMul/ReadVariableOpЂ&model_2/dense_5/BiasAdd/ReadVariableOpЂ%model_2/dense_5/MatMul/ReadVariableOpЂ&model_2/mapping_layer_2/ReadVariableOpЂ(model_2/mapping_layer_2/ReadVariableOp_1Ђ(model_2/mapping_layer_2/ReadVariableOp_2Ђ,model_2/mapping_layer_2/add_2/ReadVariableOpЂ*model_2/mapping_layer_2/sub/ReadVariableOpЂ-model_2/quality_output/BiasAdd/ReadVariableOpЂ,model_2/quality_output/MatMul/ReadVariableOpc
!model_2/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ђ
model_2/concatenate_2/concatConcatV2input_5input_6*model_2/concatenate_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:џџџџџџџџџ 
%model_2/dense_4/MatMul/ReadVariableOpReadVariableOp.model_2_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0Љ
model_2/dense_4/MatMulMatMul%model_2/concatenate_2/concat:output:0-model_2/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
&model_2/dense_4/BiasAdd/ReadVariableOpReadVariableOp/model_2_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ї
model_2/dense_4/BiasAddBiasAdd model_2/dense_4/MatMul:product:0.model_2/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџw
model_2/dense_4/SigmoidSigmoid model_2/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџv
model_2/dropout_4/IdentityIdentitymodel_2/dense_4/Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџ
%model_2/dense_5/MatMul/ReadVariableOpReadVariableOp.model_2_dense_5_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0І
model_2/dense_5/MatMulMatMul#model_2/dropout_4/Identity:output:0-model_2/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
&model_2/dense_5/BiasAdd/ReadVariableOpReadVariableOp/model_2_dense_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0І
model_2/dense_5/BiasAddBiasAdd model_2/dense_5/MatMul:product:0.model_2/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@v
model_2/dense_5/SigmoidSigmoid model_2/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@u
model_2/dropout_5/IdentityIdentitymodel_2/dense_5/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@Ђ
,model_2/quality_output/MatMul/ReadVariableOpReadVariableOp5model_2_quality_output_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Д
model_2/quality_output/MatMulMatMul#model_2/dropout_5/Identity:output:04model_2/quality_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
-model_2/quality_output/BiasAdd/ReadVariableOpReadVariableOp6model_2_quality_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Л
model_2/quality_output/BiasAddBiasAdd'model_2/quality_output/MatMul:product:05model_2/quality_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
*model_2/mapping_layer_2/sub/ReadVariableOpReadVariableOp3model_2_mapping_layer_2_sub_readvariableop_resource*
_output_shapes
: *
dtype0Б
model_2/mapping_layer_2/subSub'model_2/quality_output/BiasAdd:output:02model_2/mapping_layer_2/sub/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
&model_2/mapping_layer_2/ReadVariableOpReadVariableOp/model_2_mapping_layer_2_readvariableop_resource*
_output_shapes
: *
dtype0Ѕ
model_2/mapping_layer_2/mulMul.model_2/mapping_layer_2/ReadVariableOp:value:0model_2/mapping_layer_2/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџu
model_2/mapping_layer_2/ExpExpmodel_2/mapping_layer_2/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџb
model_2/mapping_layer_2/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model_2/mapping_layer_2/addAddV2&model_2/mapping_layer_2/add/x:output:0model_2/mapping_layer_2/Exp:y:0*
T0*'
_output_shapes
:џџџџџџџџџf
!model_2/mapping_layer_2/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Љ
model_2/mapping_layer_2/truedivRealDiv*model_2/mapping_layer_2/truediv/x:output:0model_2/mapping_layer_2/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџd
model_2/mapping_layer_2/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ѕ
model_2/mapping_layer_2/sub_1Sub(model_2/mapping_layer_2/sub_1/x:output:0#model_2/mapping_layer_2/truediv:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
(model_2/mapping_layer_2/ReadVariableOp_1ReadVariableOp1model_2_mapping_layer_2_readvariableop_1_resource*
_output_shapes
: *
dtype0Ћ
model_2/mapping_layer_2/mul_1Mul0model_2/mapping_layer_2/ReadVariableOp_1:value:0!model_2/mapping_layer_2/sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
(model_2/mapping_layer_2/ReadVariableOp_2ReadVariableOp1model_2_mapping_layer_2_readvariableop_2_resource*
_output_shapes
: *
dtype0Б
model_2/mapping_layer_2/mul_2Mul0model_2/mapping_layer_2/ReadVariableOp_2:value:0'model_2/quality_output/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
model_2/mapping_layer_2/add_1AddV2!model_2/mapping_layer_2/mul_1:z:0!model_2/mapping_layer_2/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
,model_2/mapping_layer_2/add_2/ReadVariableOpReadVariableOp5model_2_mapping_layer_2_add_2_readvariableop_resource*
_output_shapes
: *
dtype0Б
model_2/mapping_layer_2/add_2AddV2!model_2/mapping_layer_2/add_1:z:04model_2/mapping_layer_2/add_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџp
IdentityIdentity!model_2/mapping_layer_2/add_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЂ
NoOpNoOp'^model_2/dense_4/BiasAdd/ReadVariableOp&^model_2/dense_4/MatMul/ReadVariableOp'^model_2/dense_5/BiasAdd/ReadVariableOp&^model_2/dense_5/MatMul/ReadVariableOp'^model_2/mapping_layer_2/ReadVariableOp)^model_2/mapping_layer_2/ReadVariableOp_1)^model_2/mapping_layer_2/ReadVariableOp_2-^model_2/mapping_layer_2/add_2/ReadVariableOp+^model_2/mapping_layer_2/sub/ReadVariableOp.^model_2/quality_output/BiasAdd/ReadVariableOp-^model_2/quality_output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : 2P
&model_2/dense_4/BiasAdd/ReadVariableOp&model_2/dense_4/BiasAdd/ReadVariableOp2N
%model_2/dense_4/MatMul/ReadVariableOp%model_2/dense_4/MatMul/ReadVariableOp2P
&model_2/dense_5/BiasAdd/ReadVariableOp&model_2/dense_5/BiasAdd/ReadVariableOp2N
%model_2/dense_5/MatMul/ReadVariableOp%model_2/dense_5/MatMul/ReadVariableOp2P
&model_2/mapping_layer_2/ReadVariableOp&model_2/mapping_layer_2/ReadVariableOp2T
(model_2/mapping_layer_2/ReadVariableOp_1(model_2/mapping_layer_2/ReadVariableOp_12T
(model_2/mapping_layer_2/ReadVariableOp_2(model_2/mapping_layer_2/ReadVariableOp_22\
,model_2/mapping_layer_2/add_2/ReadVariableOp,model_2/mapping_layer_2/add_2/ReadVariableOp2X
*model_2/mapping_layer_2/sub/ReadVariableOp*model_2/mapping_layer_2/sub/ReadVariableOp2^
-model_2/quality_output/BiasAdd/ReadVariableOp-model_2/quality_output/BiasAdd/ReadVariableOp2\
,model_2/quality_output/MatMul/ReadVariableOp,model_2/quality_output/MatMul/ReadVariableOp:Q M
(
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_5:QM
(
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_6


c
D__inference_dropout_5_layer_call_and_return_conditional_losses_38917

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>І
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Є

і
B__inference_dense_4_layer_call_and_return_conditional_losses_38843

inputs2
matmul_readvariableop_resource:
 .
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџW
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
я
b
)__inference_dropout_5_layer_call_fn_38900

inputs
identityЂStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_38365o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs"
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ё
serving_defaultн
<
input_51
serving_default_input_5:0џџџџџџџџџ
<
input_61
serving_default_input_6:0џџџџџџџџџC
mapping_layer_20
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:ШЭ
Р
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer-4
layer_with_weights-1
layer-5
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
Ѕ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
Л
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
 bias"
_tf_keras_layer
М
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
'_random_generator"
_tf_keras_layer
Л
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

.kernel
/bias"
_tf_keras_layer
М
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6_random_generator"
_tf_keras_layer
Л
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses

=kernel
>bias"
_tf_keras_layer
с
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses

Eparam1

Fparam2

Gparam3

Hparam4

Iparam5"
_tf_keras_layer
n
0
 1
.2
/3
=4
>5
E6
F7
G8
H9
I10"
trackable_list_wrapper
n
0
 1
.2
/3
=4
>5
E6
F7
G8
H9
I10"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
б
Otrace_0
Ptrace_1
Qtrace_2
Rtrace_32ц
'__inference_model_2_layer_call_fn_38319
'__inference_model_2_layer_call_fn_38664
'__inference_model_2_layer_call_fn_38692
'__inference_model_2_layer_call_fn_38536П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zOtrace_0zPtrace_1zQtrace_2zRtrace_3
Н
Strace_0
Ttrace_1
Utrace_2
Vtrace_32в
B__inference_model_2_layer_call_and_return_conditional_losses_38744
B__inference_model_2_layer_call_and_return_conditional_losses_38810
B__inference_model_2_layer_call_and_return_conditional_losses_38570
B__inference_model_2_layer_call_and_return_conditional_losses_38604П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zStrace_0zTtrace_1zUtrace_2zVtrace_3
дBб
 __inference__wrapped_model_38172input_5input_6"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 

W
_variables
X_iterations
Y_learning_rate
Z_index_dict
[
_momentums
\_velocities
]_update_step_xla"
experimentalOptimizer
,
^serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ё
dtrace_02д
-__inference_concatenate_2_layer_call_fn_38816Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zdtrace_0

etrace_02я
H__inference_concatenate_2_layer_call_and_return_conditional_losses_38823Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zetrace_0
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ы
ktrace_02Ю
'__inference_dense_4_layer_call_fn_38832Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zktrace_0

ltrace_02щ
B__inference_dense_4_layer_call_and_return_conditional_losses_38843Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zltrace_0
": 
 2dense_4/kernel
:2dense_4/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
У
rtrace_0
strace_12
)__inference_dropout_4_layer_call_fn_38848
)__inference_dropout_4_layer_call_fn_38853Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zrtrace_0zstrace_1
љ
ttrace_0
utrace_12Т
D__inference_dropout_4_layer_call_and_return_conditional_losses_38858
D__inference_dropout_4_layer_call_and_return_conditional_losses_38870Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zttrace_0zutrace_1
"
_generic_user_object
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
ы
{trace_02Ю
'__inference_dense_5_layer_call_fn_38879Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z{trace_0

|trace_02щ
B__inference_dense_5_layer_call_and_return_conditional_losses_38890Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z|trace_0
!:	@2dense_5/kernel
:@2dense_5/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Џ
}non_trainable_variables

~layers
metrics
 layer_regularization_losses
layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
Ч
trace_0
trace_12
)__inference_dropout_5_layer_call_fn_38895
)__inference_dropout_5_layer_call_fn_38900Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
§
trace_0
trace_12Т
D__inference_dropout_5_layer_call_and_return_conditional_losses_38905
D__inference_dropout_5_layer_call_and_return_conditional_losses_38917Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
"
_generic_user_object
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
є
trace_02е
.__inference_quality_output_layer_call_fn_38926Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02№
I__inference_quality_output_layer_call_and_return_conditional_losses_38936Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
':%@2quality_output/kernel
!:2quality_output/bias
C
E0
F1
G2
H3
I4"
trackable_list_wrapper
C
E0
F1
G2
H3
I4"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
№
trace_02б
/__inference_mapping_layer_2_layer_call_fn_38951
В
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02ь
J__inference_mapping_layer_2_layer_call_and_return_conditional_losses_38978
В
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
: 2Variable
: 2Variable
: 2Variable
: 2Variable
: 2Variable
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Bџ
'__inference_model_2_layer_call_fn_38319input_5input_6"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
'__inference_model_2_layer_call_fn_38664inputs_0inputs_1"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
'__inference_model_2_layer_call_fn_38692inputs_0inputs_1"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Bџ
'__inference_model_2_layer_call_fn_38536input_5input_6"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
B__inference_model_2_layer_call_and_return_conditional_losses_38744inputs_0inputs_1"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
B__inference_model_2_layer_call_and_return_conditional_losses_38810inputs_0inputs_1"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
B__inference_model_2_layer_call_and_return_conditional_losses_38570input_5input_6"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
B__inference_model_2_layer_call_and_return_conditional_losses_38604input_5input_6"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ф
X0
1
2
3
4
5
6
7
8
9
10
11
 12
Ё13
Ђ14
Ѓ15
Є16
Ѕ17
І18
Ї19
Ј20
Љ21
Њ22"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
y
0
1
2
3
4
5
Ё6
Ѓ7
Ѕ8
Ї9
Љ10"
trackable_list_wrapper
y
0
1
2
3
4
 5
Ђ6
Є7
І8
Ј9
Њ10"
trackable_list_wrapper
П2МЙ
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
бBЮ
#__inference_signature_wrapper_38636input_5input_6"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
эBъ
-__inference_concatenate_2_layer_call_fn_38816inputs_0inputs_1"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
H__inference_concatenate_2_layer_call_and_return_conditional_losses_38823inputs_0inputs_1"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
лBи
'__inference_dense_4_layer_call_fn_38832inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
іBѓ
B__inference_dense_4_layer_call_and_return_conditional_losses_38843inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
юBы
)__inference_dropout_4_layer_call_fn_38848inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
)__inference_dropout_4_layer_call_fn_38853inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_dropout_4_layer_call_and_return_conditional_losses_38858inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_dropout_4_layer_call_and_return_conditional_losses_38870inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
лBи
'__inference_dense_5_layer_call_fn_38879inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
іBѓ
B__inference_dense_5_layer_call_and_return_conditional_losses_38890inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
юBы
)__inference_dropout_5_layer_call_fn_38895inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
)__inference_dropout_5_layer_call_fn_38900inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_dropout_5_layer_call_and_return_conditional_losses_38905inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_dropout_5_layer_call_and_return_conditional_losses_38917inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
тBп
.__inference_quality_output_layer_call_fn_38926inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
§Bњ
I__inference_quality_output_layer_call_and_return_conditional_losses_38936inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
йBж
/__inference_mapping_layer_2_layer_call_fn_38951x"
В
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
єBё
J__inference_mapping_layer_2_layer_call_and_return_conditional_losses_38978x"
В
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
R
Ћ	variables
Ќ	keras_api

­total

Ўcount"
_tf_keras_metric
':%
 2Adam/m/dense_4/kernel
':%
 2Adam/v/dense_4/kernel
 :2Adam/m/dense_4/bias
 :2Adam/v/dense_4/bias
&:$	@2Adam/m/dense_5/kernel
&:$	@2Adam/v/dense_5/kernel
:@2Adam/m/dense_5/bias
:@2Adam/v/dense_5/bias
,:*@2Adam/m/quality_output/kernel
,:*@2Adam/v/quality_output/kernel
&:$2Adam/m/quality_output/bias
&:$2Adam/v/quality_output/bias
: 2Adam/m/Variable
: 2Adam/v/Variable
: 2Adam/m/Variable
: 2Adam/v/Variable
: 2Adam/m/Variable
: 2Adam/v/Variable
: 2Adam/m/Variable
: 2Adam/v/Variable
: 2Adam/m/Variable
: 2Adam/v/Variable
0
­0
Ў1"
trackable_list_wrapper
.
Ћ	variables"
_generic_user_object
:  (2total
:  (2countб
 __inference__wrapped_model_38172Ќ ./=>GFEHIZЂW
PЂM
KH
"
input_5џџџџџџџџџ
"
input_6џџџџџџџџџ
Њ "AЊ>
<
mapping_layer_2)&
mapping_layer_2џџџџџџџџџк
H__inference_concatenate_2_layer_call_and_return_conditional_losses_38823\ЂY
RЂO
MJ
# 
inputs_0џџџџџџџџџ
# 
inputs_1џџџџџџџџџ
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ 
 Д
-__inference_concatenate_2_layer_call_fn_38816\ЂY
RЂO
MJ
# 
inputs_0џџџџџџџџџ
# 
inputs_1џџџџџџџџџ
Њ ""
unknownџџџџџџџџџ Ћ
B__inference_dense_4_layer_call_and_return_conditional_losses_38843e 0Ђ-
&Ђ#
!
inputsџџџџџџџџџ 
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ
 
'__inference_dense_4_layer_call_fn_38832Z 0Ђ-
&Ђ#
!
inputsџџџџџџџџџ 
Њ ""
unknownџџџџџџџџџЊ
B__inference_dense_5_layer_call_and_return_conditional_losses_38890d./0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ@
 
'__inference_dense_5_layer_call_fn_38879Y./0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "!
unknownџџџџџџџџџ@­
D__inference_dropout_4_layer_call_and_return_conditional_losses_38858e4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ
 ­
D__inference_dropout_4_layer_call_and_return_conditional_losses_38870e4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ
 
)__inference_dropout_4_layer_call_fn_38848Z4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ ""
unknownџџџџџџџџџ
)__inference_dropout_4_layer_call_fn_38853Z4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ ""
unknownџџџџџџџџџЋ
D__inference_dropout_5_layer_call_and_return_conditional_losses_38905c3Ђ0
)Ђ&
 
inputsџџџџџџџџџ@
p 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ@
 Ћ
D__inference_dropout_5_layer_call_and_return_conditional_losses_38917c3Ђ0
)Ђ&
 
inputsџџџџџџџџџ@
p
Њ ",Ђ)
"
tensor_0џџџџџџџџџ@
 
)__inference_dropout_5_layer_call_fn_38895X3Ђ0
)Ђ&
 
inputsџџџџџџџџџ@
p 
Њ "!
unknownџџџџџџџџџ@
)__inference_dropout_5_layer_call_fn_38900X3Ђ0
)Ђ&
 
inputsџџџџџџџџџ@
p
Њ "!
unknownџџџџџџџџџ@Џ
J__inference_mapping_layer_2_layer_call_and_return_conditional_losses_38978aGFEHI*Ђ'
 Ђ

xџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
/__inference_mapping_layer_2_layer_call_fn_38951VGFEHI*Ђ'
 Ђ

xџџџџџџџџџ
Њ "!
unknownџџџџџџџџџц
B__inference_model_2_layer_call_and_return_conditional_losses_38570 ./=>GFEHIbЂ_
XЂU
KH
"
input_5џџџџџџџџџ
"
input_6џџџџџџџџџ
p 

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 ц
B__inference_model_2_layer_call_and_return_conditional_losses_38604 ./=>GFEHIbЂ_
XЂU
KH
"
input_5џџџџџџџџџ
"
input_6џџџџџџџџџ
p

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 ш
B__inference_model_2_layer_call_and_return_conditional_losses_38744Ё ./=>GFEHIdЂa
ZЂW
MJ
# 
inputs_0џџџџџџџџџ
# 
inputs_1џџџџџџџџџ
p 

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 ш
B__inference_model_2_layer_call_and_return_conditional_losses_38810Ё ./=>GFEHIdЂa
ZЂW
MJ
# 
inputs_0џџџџџџџџџ
# 
inputs_1џџџџџџџџџ
p

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 Р
'__inference_model_2_layer_call_fn_38319 ./=>GFEHIbЂ_
XЂU
KH
"
input_5џџџџџџџџџ
"
input_6џџџџџџџџџ
p 

 
Њ "!
unknownџџџџџџџџџР
'__inference_model_2_layer_call_fn_38536 ./=>GFEHIbЂ_
XЂU
KH
"
input_5џџџџџџџџџ
"
input_6џџџџџџџџџ
p

 
Њ "!
unknownџџџџџџџџџТ
'__inference_model_2_layer_call_fn_38664 ./=>GFEHIdЂa
ZЂW
MJ
# 
inputs_0џџџџџџџџџ
# 
inputs_1џџџџџџџџџ
p 

 
Њ "!
unknownџџџџџџџџџТ
'__inference_model_2_layer_call_fn_38692 ./=>GFEHIdЂa
ZЂW
MJ
# 
inputs_0џџџџџџџџџ
# 
inputs_1џџџџџџџџџ
p

 
Њ "!
unknownџџџџџџџџџА
I__inference_quality_output_layer_call_and_return_conditional_losses_38936c=>/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
.__inference_quality_output_layer_call_fn_38926X=>/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "!
unknownџџџџџџџџџх
#__inference_signature_wrapper_38636Н ./=>GFEHIkЂh
Ђ 
aЊ^
-
input_5"
input_5џџџџџџџџџ
-
input_6"
input_6џџџџџџџџџ"AЊ>
<
mapping_layer_2)&
mapping_layer_2џџџџџџџџџ