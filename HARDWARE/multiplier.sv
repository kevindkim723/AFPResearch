module afp_multiplier(
    input logic [3:0] x, y,
    output logic [3:0] result);






endmodule

module unpack(
    input logic [3:0] x,
    output logic xs, [1:0] xm, [1:0] xo
);

logic [1:0] xm_in   ;
logic denorm;

assign xs = x[3];
assign xo = x[2:1];
assign denorm = (xo == 2'b11);
assign xm_in = x[0];
assign xm = (denorm) ? {xm_in, 1'b0} : {1'b1, xm_in};


endmodule

module multiplier(
    input logic [1:0] xm,ym,
    output logic [3:0] pm
);
logic [3:0] xmym;
assign xmym = {xm,ym};
always_comb
    case (xmym)
        4'b1111: pm = 4'b1001;
        4'b1010: pm = 4'b0100;
        4'b1011: pm = 4'b0110;
        4'b1110: pm = 4'b0110;
        default: pm = 4'b0;
    endcase

endmodule

module adder(
    input logic [2:0] yo,xo;
    output logic [3:0] po;
);

assign po = yo+xo;

endmodule

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
assign pm_norm = (pm[3]) ? 0 : pm[1]
assign po_norm = 

endmodule

