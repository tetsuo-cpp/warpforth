//===- ForthDialect.cpp - Forth dialect ----------------------------------===//
//
// This file implements the Forth dialect.
//
//===----------------------------------------------------------------------===//

#include "warpforth/Dialect/Forth/ForthDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::forth;

#include "warpforth/Dialect/Forth/ForthOpsDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "warpforth/Dialect/Forth/ForthOpsTypes.cpp.inc"

#define GET_OP_CLASSES
#include "warpforth/Dialect/Forth/ForthOps.cpp.inc"

//===----------------------------------------------------------------------===//
// IfOp RegionBranchOpInterface.
//===----------------------------------------------------------------------===//

void IfOp::getSuccessorRegions(RegionBranchPoint point,
                               SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent()) {
    // From parent: branch into then or else region.
    regions.push_back(
        RegionSuccessor(&getThenRegion(), getThenRegion().getArguments()));
    regions.push_back(
        RegionSuccessor(&getElseRegion(), getElseRegion().getArguments()));
    return;
  }
  // From either region: return to parent with op results.
  regions.push_back(RegionSuccessor(getOperation()->getResults()));
}

OperandRange IfOp::getEntrySuccessorOperands(RegionBranchPoint point) {
  return getOperation()->getOperands();
}

//===----------------------------------------------------------------------===//
// IfOp custom assembly format.
//===----------------------------------------------------------------------===//

void IfOp::print(OpAsmPrinter &p) {
  p << ' ' << getInputStack() << " : " << getInputStack().getType() << " -> "
    << getOutputStack().getType() << ' ';
  p.printRegion(getThenRegion());
  p << " else ";
  p.printRegion(getElseRegion());
}

ParseResult IfOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand inputStack;
  Type inputType, outputType;

  if (parser.parseOperand(inputStack) || parser.parseColon() ||
      parser.parseType(inputType) || parser.parseArrow() ||
      parser.parseType(outputType) ||
      parser.resolveOperand(inputStack, inputType, result.operands))
    return failure();

  result.addTypes(outputType);

  // Parse then region.
  auto *thenRegion = result.addRegion();
  if (parser.parseRegion(*thenRegion))
    return failure();

  // Parse "else" keyword and else region.
  if (parser.parseKeyword("else"))
    return failure();

  auto *elseRegion = result.addRegion();
  if (parser.parseRegion(*elseRegion))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// BeginUntilOp RegionBranchOpInterface.
//===----------------------------------------------------------------------===//

void BeginUntilOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent()) {
    // From parent: enter the body region.
    regions.push_back(
        RegionSuccessor(&getBodyRegion(), getBodyRegion().getArguments()));
    return;
  }
  // From body: loop back to body or exit to parent.
  regions.push_back(
      RegionSuccessor(&getBodyRegion(), getBodyRegion().getArguments()));
  regions.push_back(RegionSuccessor(getOperation()->getResults()));
}

OperandRange BeginUntilOp::getEntrySuccessorOperands(RegionBranchPoint point) {
  return getOperation()->getOperands();
}

//===----------------------------------------------------------------------===//
// BeginUntilOp custom assembly format.
//===----------------------------------------------------------------------===//

void BeginUntilOp::print(OpAsmPrinter &p) {
  p << ' ' << getInputStack() << " : " << getInputStack().getType() << " -> "
    << getOutputStack().getType() << ' ';
  p.printRegion(getBodyRegion());
}

ParseResult BeginUntilOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand inputStack;
  Type inputType, outputType;

  if (parser.parseOperand(inputStack) || parser.parseColon() ||
      parser.parseType(inputType) || parser.parseArrow() ||
      parser.parseType(outputType) ||
      parser.resolveOperand(inputStack, inputType, result.operands))
    return failure();

  result.addTypes(outputType);

  auto *bodyRegion = result.addRegion();
  if (parser.parseRegion(*bodyRegion))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// BeginWhileRepeatOp RegionBranchOpInterface.
//===----------------------------------------------------------------------===//

void BeginWhileRepeatOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent()) {
    // From parent: enter the condition region.
    regions.push_back(RegionSuccessor(&getConditionRegion(),
                                      getConditionRegion().getArguments()));
    return;
  }
  if (point.getRegionOrNull() == &getConditionRegion()) {
    // From condition: enter body or exit to parent.
    regions.push_back(
        RegionSuccessor(&getBodyRegion(), getBodyRegion().getArguments()));
    regions.push_back(RegionSuccessor(getOperation()->getResults()));
    return;
  }
  // From body: loop back to condition.
  regions.push_back(RegionSuccessor(&getConditionRegion(),
                                    getConditionRegion().getArguments()));
}

OperandRange
BeginWhileRepeatOp::getEntrySuccessorOperands(RegionBranchPoint point) {
  return getOperation()->getOperands();
}

//===----------------------------------------------------------------------===//
// BeginWhileRepeatOp custom assembly format.
//===----------------------------------------------------------------------===//

void BeginWhileRepeatOp::print(OpAsmPrinter &p) {
  p << ' ' << getInputStack() << " : " << getInputStack().getType() << " -> "
    << getOutputStack().getType() << ' ';
  p.printRegion(getConditionRegion());
  p << " do ";
  p.printRegion(getBodyRegion());
}

ParseResult BeginWhileRepeatOp::parse(OpAsmParser &parser,
                                      OperationState &result) {
  OpAsmParser::UnresolvedOperand inputStack;
  Type inputType, outputType;

  if (parser.parseOperand(inputStack) || parser.parseColon() ||
      parser.parseType(inputType) || parser.parseArrow() ||
      parser.parseType(outputType) ||
      parser.resolveOperand(inputStack, inputType, result.operands))
    return failure();

  result.addTypes(outputType);

  // Parse condition region.
  auto *condRegion = result.addRegion();
  if (parser.parseRegion(*condRegion))
    return failure();

  // Parse "do" keyword and body region.
  if (parser.parseKeyword("do"))
    return failure();

  auto *bodyRegion = result.addRegion();
  if (parser.parseRegion(*bodyRegion))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// DoLoopOp RegionBranchOpInterface.
//===----------------------------------------------------------------------===//

void DoLoopOp::getSuccessorRegions(RegionBranchPoint point,
                                   SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent()) {
    // From parent: enter the body region.
    regions.push_back(
        RegionSuccessor(&getBodyRegion(), getBodyRegion().getArguments()));
    return;
  }
  // From body: loop back to body or exit to parent.
  regions.push_back(
      RegionSuccessor(&getBodyRegion(), getBodyRegion().getArguments()));
  regions.push_back(RegionSuccessor(getOperation()->getResults()));
}

OperandRange DoLoopOp::getEntrySuccessorOperands(RegionBranchPoint point) {
  return getOperation()->getOperands();
}

//===----------------------------------------------------------------------===//
// DoLoopOp custom assembly format.
//===----------------------------------------------------------------------===//

void DoLoopOp::print(OpAsmPrinter &p) {
  p << ' ' << getInputStack() << " : " << getInputStack().getType() << " -> "
    << getOutputStack().getType() << ' ';
  p.printRegion(getBodyRegion());
}

ParseResult DoLoopOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand inputStack;
  Type inputType, outputType;

  if (parser.parseOperand(inputStack) || parser.parseColon() ||
      parser.parseType(inputType) || parser.parseArrow() ||
      parser.parseType(outputType) ||
      parser.resolveOperand(inputStack, inputType, result.operands))
    return failure();

  result.addTypes(outputType);

  auto *bodyRegion = result.addRegion();
  if (parser.parseRegion(*bodyRegion))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// Forth dialect.
//===----------------------------------------------------------------------===//

void ForthDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "warpforth/Dialect/Forth/ForthOpsTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "warpforth/Dialect/Forth/ForthOps.cpp.inc"
      >();
}
