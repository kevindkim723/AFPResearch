module afp_multiplier(
    input logic [4:0] x, y,
    output logic [3:0] pm,
    output logic [3:0] po,
    output logic ps);

    logic xs, ys;
    logic [1:0] xm, ym;
    logic [2:0] xo, yo;
    assign ps = xs ^ ys;
    unpack UnpackX(.x(x), .xs(xs), .xm(xm), .xo(xo));
    unpack UnpackY(.x(y), .xs(ys), .xm(ym), .xo(yo));
    multiplier Mult(.xm(xm), .ym(ym), .pm(pm));
    adder Add(.yo(yo), .xo(xo), .po(po));
endmodule

module unpack(
    input logic [4:0] x,
    output logic xs, [1:0] xm, [2:0] xo
);

logic [1:0] xm_in;
logic denorm;

assign xs = x[3];
assign xo = x[3:1];
assign denorm = (xo == 3'b111);
assign xm_in = x[0];
assign xm = (denorm) ? {xm_in, 1'b0} : {1'b1, xm_in};

endmodule

module multiplier(
    input logic [1:0] xm,ym,
    output logic [3:0] pm
);

assign a1 = xm[1] & ym[1];
assign a2 = &xm;
assign a3 = xm[0]^ym[0];
assign a4 = xm[0] & ym[0];
assign a5 = a1&a2;
assign a6 = a1&a3;
assign a7 = !a4 & a1;

assign pm = {a5, a7, a6, a5};


endmodule

module adder(
    input logic [2:0] yo,xo,
    output logic [3:0] po
);

assign po = yo+xo;

endmodule
/*
module normalize(
    input logic [3:0] pm, po,
    output logic pm_norm,
    output logic [1:0] po_norm
);
logic [4:0] offsetSub3;
logic [3:0] leadingZeros;

always_comb
    casez (pm)
        4'b1???: leadingZeros = 4'b0000;
        4'b01??: leadingZeros = 4'b0001;
        4'b001?: leadingZeros = 4'b0010;
        4'b0001: leadingZeros = 4'b0011;
        4'b0000: leadingZeros = 4'b0100;
    endcase

assign offsetSub3 = {1'b0, po} + ~5'b00011 + 1;

always_comb 
    case (po)
    4'b
//assign pm_norm = (pm[3]) ? 0 : pm[1]
//assign po_norm = 

endmodule
*/