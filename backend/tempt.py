#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedded PTX static analyzer test.
PTX code is hardcoded in this file for quick development.
"""

from lark import Lark, UnexpectedInput

# --- LARK Grammar ---
GRAMMAR = r"""
%import common.WS
%import common.NEWLINE
%ignore WS
%ignore NEWLINE

COMMENT: "//" /[^\n]*/

REG: /%[a-zA-Z_][a-zA-Z0-9_]*/
FIELD: /\.(?:[xyz]|lo|hi)/
IMM_HEX: /0[fFdD][0-9a-fA-F]{8}/  
IMM:  /-?\d+(?:\.\d+)?(?![A-Za-z])/
MEM: /\[[^\]]+\]/
LABEL: /\$[a-zA-Z0-9_]+/
IDENT: /[a-zA-Z_][a-zA-Z0-9_.]*/    
RBRACE: "}"         
SEMICOLON: ";"
COMMA: ","

%ignore RBRACE

// --- New part: skip .entry declaration entirely
skip_entry: ENTRY_LINE
ENTRY_LINE: /.visible\s+\.entry(.|\n)*?\{/

?start: skip_entry? block_body
block_body: line+

line: instruction SEMICOLON         -> instr_line
    | label_def                     -> label_line
    | COMMENT                       -> comment_line

label_def: LABEL ":"                -> label_def

instruction: pred? opcode operand_list?   -> instruction

pred: "@" REG                       -> pred
opcode: IDENT                       -> opcode

operand_list: operand (COMMA operand)*
operand: REG FIELD?
       | IMM_HEX
       | IMM
       | MEM
       | LABEL
       | IDENT
"""

# --- Sample PTX code ---

PTX_SAMPLE = """
.visible .entry matmul_kernel(float const*, float const*, float*, int)(
        .param .u64 matmul_kernel(float const*, float const*, float*, int)_param_0,
        .param .u64 matmul_kernel(float const*, float const*, float*, int)_param_1,
        .param .u64 matmul_kernel(float const*, float const*, float*, int)_param_2,
        .param .u32 matmul_kernel(float const*, float const*, float*, int)_param_3
)
{

        ld.param.u64    %rd18, [matmul_kernel(float const*, float const*, float*, int)_param_0];
        ld.param.u64    %rd19, [matmul_kernel(float const*, float const*, float*, int)_param_1];
        ld.param.u64    %rd17, [matmul_kernel(float const*, float const*, float*, int)_param_2];
        ld.param.u32    %r13, [matmul_kernel(float const*, float const*, float*, int)_param_3];
        cvta.to.global.u64      %rd1, %rd19;
        cvta.to.global.u64      %rd2, %rd18;
        mov.u32         %r14, %ntid.y;
        mov.u32         %r15, %ctaid.y;
        mov.u32         %r16, %tid.y;
        mad.lo.s32      %r1, %r15, %r14, %r16;
        mov.u32         %r17, %ntid.x;
        mov.u32         %r18, %ctaid.x;
        mov.u32         %r19, %tid.x;
        mad.lo.s32      %r2, %r18, %r17, %r19;
        setp.ge.s32     %p1, %r1, %r13;
        setp.ge.s32     %p2, %r2, %r13;
        or.pred         %p3, %p1, %p2;
        @%p3 bra        $L__BB0_9;

        setp.lt.s32     %p4, %r13, 1;
        mul.lo.s32      %r3, %r1, %r13;
        mov.f32         %f29, 0f00000000;
        @%p4 bra        $L__BB0_8;

        add.s32         %r21, %r13, -1;
        and.b32         %r29, %r13, 3;
        setp.lt.u32     %p5, %r21, 3;
        mov.f32         %f29, 0f00000000;
        mov.u32         %r28, 0;
        @%p5 bra        $L__BB0_5;

        sub.s32         %r27, %r13, %r29;
        mul.wide.s32    %rd3, %r3, 4;
        mul.wide.s32    %rd20, %r2, 4;
        add.s64         %rd30, %rd1, %rd20;
        mul.wide.s32    %rd5, %r13, 4;
        mov.f32         %f29, 0f00000000;
        mov.u32         %r28, 0;
        mov.u64         %rd31, %rd2;

$L__BB0_4:
        add.s64         %rd21, %rd31, %rd3;
        ld.global.f32   %f12, [%rd30];
        ld.global.f32   %f13, [%rd21];
        fma.rn.f32      %f14, %f13, %f12, %f29;
        add.s64         %rd22, %rd30, %rd5;
        ld.global.f32   %f15, [%rd22];
        ld.global.f32   %f16, [%rd21+4];
        fma.rn.f32      %f17, %f16, %f15, %f14;
        add.s64         %rd23, %rd22, %rd5;
        ld.global.f32   %f18, [%rd23];
        ld.global.f32   %f19, [%rd21+8];
        fma.rn.f32      %f20, %f19, %f18, %f17;
        add.s64         %rd24, %rd23, %rd5;
        add.s64         %rd30, %rd24, %rd5;
        ld.global.f32   %f21, [%rd24];
        ld.global.f32   %f22, [%rd21+12];
        fma.rn.f32      %f29, %f22, %f21, %f20;
        add.s32         %r28, %r28, 4;
        add.s64         %rd31, %rd31, 16;
        add.s32         %r27, %r27, -4;
        setp.ne.s32     %p6, %r27, 0;
        @%p6 bra        $L__BB0_4;

$L__BB0_5:
        setp.eq.s32     %p7, %r29, 0;
        @%p7 bra        $L__BB0_8;

        mad.lo.s32      %r23, %r28, %r13, %r2;
        mul.wide.s32    %rd25, %r23, 4;
        add.s64         %rd33, %rd1, %rd25;
        mul.wide.s32    %rd11, %r13, 4;
        add.s32         %r24, %r28, %r3;
        mul.wide.s32    %rd26, %r24, 4;
        add.s64         %rd32, %rd2, %rd26;

$L__BB0_7:
        ld.global.f32   %f23, [%rd33];
        ld.global.f32   %f24, [%rd32];
        fma.rn.f32      %f29, %f24, %f23, %f29;
        add.s64         %rd33, %rd33, %rd11;
        add.s64         %rd32, %rd32, 4;
        add.s32         %r29, %r29, -1;
        setp.ne.s32     %p8, %r29, 0;
        @%p8 bra        $L__BB0_7;

$L__BB0_8:
        add.s32         %r25, %r3, %r2;
        cvta.to.global.u64      %rd27, %rd17;
        mul.wide.s32    %rd28, %r25, 4;
        add.s64         %rd29, %rd27, %rd28;
        st.global.f32   [%rd29], %f29;

$L__BB0_9:
        ret;

}
"""
# --- Main Function ---
def main():
    parser = Lark(GRAMMAR, parser="lalr", lexer="basic", propagate_positions=True)

    try:
        tree = parser.parse(PTX_SAMPLE)
        print(tree)  # Print AST
    except UnexpectedInput as e:
        line, col = e.line, e.column
        error_line = PTX_SAMPLE.splitlines()[line - 1]
        pointer = " " * (col - 1) + "^"
        print(f"\nParse error at line {line}, col {col}:\n{error_line}\n{pointer}")
        print(str(e))

if __name__ == "__main__":
    main()
