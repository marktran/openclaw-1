import type { Node as TreeSitterNode } from "web-tree-sitter";
import { unwrapKnownDispatchWrapperInvocation } from "../dispatch-wrapper-resolution.js";
import { detectInterpreterInlineEvalArgv } from "../exec-inline-eval.js";
import { normalizeExecutableToken } from "../exec-wrapper-resolution.js";
import {
  extractShellWrapperCommand,
  isShellWrapperExecutable,
  resolveShellWrapperTransportArgv,
} from "../shell-wrapper-resolution.js";
import { parseBashForCommandExplanation } from "./tree-sitter-runtime.js";
import type {
  CommandContext,
  CommandExplanation,
  CommandRisk,
  CommandShape,
  CommandStep,
  SourceSpan,
} from "./types.js";

type MutableExplanation = {
  shapes: Set<CommandShape>;
  commands: CommandStep[];
  risks: CommandRisk[];
};

type DynamicArgument = {
  index: number;
  text: string;
  span: SourceSpan;
};

type CommandArgv = {
  argv: string[];
  dynamicArguments: DynamicArgument[];
};

const SHELL_CARRIER_EXECUTABLES = new Set(["sudo", "doas", "env", "command", "builtin"]);
const SOURCE_EXECUTABLES = new Set([".", "source"]);

function children(node: TreeSitterNode): TreeSitterNode[] {
  return Array.from({ length: node.childCount }, (_, index) => node.child(index)).filter(
    (child): child is TreeSitterNode => child !== null,
  );
}

function namedChildren(node: TreeSitterNode): TreeSitterNode[] {
  return Array.from({ length: node.namedChildCount }, (_, index) => node.namedChild(index)).filter(
    (child): child is TreeSitterNode => child !== null,
  );
}

function hasDirectChildType(node: TreeSitterNode, type: string): boolean {
  return children(node).some((child) => child.type === type);
}

function spanFromNode(node: TreeSitterNode): SourceSpan {
  return {
    startIndex: node.startIndex,
    endIndex: node.endIndex,
    startPosition: { row: node.startPosition.row, column: node.startPosition.column },
    endPosition: { row: node.endPosition.row, column: node.endPosition.column },
  };
}

type ShellWordValue = { kind: "literal"; value: string } | { kind: "dynamic"; value: string };

const DYNAMIC_WORD_NODE_TYPES = new Set([
  "arithmetic_expansion",
  "command_substitution",
  "expansion",
  "process_substitution",
  "simple_expansion",
]);

const COMMAND_ARGUMENT_NODE_TYPES = new Set([
  "ansi_c_string",
  "arithmetic_expansion",
  "command_substitution",
  "concatenation",
  "expansion",
  "number",
  "process_substitution",
  "raw_string",
  "simple_expansion",
  "string",
  "word",
]);

function hasEscapedLineContinuation(text: string): boolean {
  return /\\(?:\r\n|[\r\n])/.test(text);
}

function hasExecutableLineContinuation(text: string): boolean {
  return /^[^\s]*\\(?:\r\n|[\r\n])/.test(text);
}

function hasUnescapedDynamicPattern(text: string): boolean {
  for (let index = 0; index < text.length; index += 1) {
    const ch = text[index];
    if (ch === "\\") {
      index += 1;
      continue;
    }
    if (ch === "*" || ch === "?") {
      return true;
    }
    if (ch === "[" && text.indexOf("]", index + 1) > index + 1) {
      return true;
    }
    if (ch === "{" && text.indexOf("}", index + 1) > index + 1) {
      return true;
    }
  }
  return false;
}

function decodeUnquotedShellText(text: string): string {
  let output = "";
  for (let index = 0; index < text.length; index += 1) {
    const ch = text[index];
    const next = text[index + 1];
    if (ch === "\\" && next !== undefined) {
      if (next === "\r" && text[index + 2] === "\n") {
        index += 2;
        continue;
      }
      if (next === "\n" || next === "\r") {
        index += 1;
        continue;
      }
      output += next;
      index += 1;
      continue;
    }
    output += ch;
  }
  return output;
}

function decodeDoubleQuotedText(text: string): string {
  const body = text.startsWith('"') && text.endsWith('"') ? text.slice(1, -1) : text;
  let output = "";
  for (let index = 0; index < body.length; index += 1) {
    const ch = body[index];
    const next = body[index + 1];
    if (ch === "\\" && next !== undefined) {
      if (next === "\r" && body[index + 2] === "\n") {
        index += 2;
        continue;
      }
      if (["\\", '"', "$", "`", "\n", "\r"].includes(next)) {
        if (next !== "\n" && next !== "\r") {
          output += next;
        }
        index += 1;
        continue;
      }
    }
    output += ch;
  }
  return output;
}

function decodeAnsiCString(text: string): string {
  const body = text.startsWith("$'") && text.endsWith("'") ? text.slice(2, -1) : text;
  let output = "";
  for (let index = 0; index < body.length; index += 1) {
    const ch = body[index];
    if (ch !== "\\") {
      output += ch;
      continue;
    }

    const next = body[index + 1];
    if (next === undefined) {
      output += "\\";
      continue;
    }

    const simpleEscapes: Record<string, string> = {
      "'": "'",
      '"': '"',
      "?": "?",
      "\\": "\\",
      a: "\u0007",
      b: "\b",
      e: "\u001B",
      E: "\u001B",
      f: "\f",
      n: "\n",
      r: "\r",
      t: "\t",
      v: "\v",
    };
    const simple = simpleEscapes[next];
    if (simple !== undefined) {
      output += simple;
      index += 1;
      continue;
    }

    if (next === "x") {
      const hex = body.slice(index + 2).match(/^[0-9A-Fa-f]{1,2}/)?.[0] ?? "";
      if (hex) {
        output += String.fromCodePoint(Number.parseInt(hex, 16));
        index += 1 + hex.length;
        continue;
      }
    }

    if (next === "u" || next === "U") {
      const maxLength = next === "u" ? 4 : 8;
      const hex =
        body.slice(index + 2).match(new RegExp(`^[0-9A-Fa-f]{1,${maxLength}}`))?.[0] ?? "";
      if (hex) {
        const codePoint = Number.parseInt(hex, 16);
        try {
          output += String.fromCodePoint(codePoint);
        } catch {
          output += `\\${next}${hex}`;
        }
        index += 1 + hex.length;
        continue;
      }
    }

    if (/^[0-7]$/.test(next)) {
      const octal = body.slice(index + 1).match(/^[0-7]{1,3}/)?.[0] ?? "";
      if (octal) {
        output += String.fromCodePoint(Number.parseInt(octal, 8));
        index += octal.length;
        continue;
      }
    }

    output += next;
    index += 1;
  }
  return output;
}

function hasDynamicWordPart(node: TreeSitterNode): boolean {
  return (
    DYNAMIC_WORD_NODE_TYPES.has(node.type) ||
    namedChildren(node).some((child) => hasDynamicWordPart(child))
  );
}

function shellWordValue(node: TreeSitterNode): ShellWordValue {
  if (DYNAMIC_WORD_NODE_TYPES.has(node.type)) {
    return { kind: "dynamic", value: node.text };
  }
  if (
    node.type !== "command_name" &&
    node.type !== "concatenation" &&
    namedChildren(node).some((child) => hasDynamicWordPart(child))
  ) {
    return {
      kind: "dynamic",
      value: node.type === "string" ? decodeDoubleQuotedText(node.text) : node.text,
    };
  }

  switch (node.type) {
    case "command_name": {
      const parts = namedChildren(node);
      if (parts.length === 0) {
        return hasUnescapedDynamicPattern(node.text)
          ? { kind: "dynamic", value: decodeUnquotedShellText(node.text) }
          : { kind: "literal", value: decodeUnquotedShellText(node.text) };
      }
      let value = "";
      for (const part of parts) {
        const partValue = shellWordValue(part);
        value += partValue.value;
        if (partValue.kind !== "literal") {
          return { kind: "dynamic", value };
        }
      }
      return { kind: "literal", value };
    }
    case "word":
      return hasUnescapedDynamicPattern(node.text)
        ? { kind: "dynamic", value: decodeUnquotedShellText(node.text) }
        : { kind: "literal", value: decodeUnquotedShellText(node.text) };
    case "raw_string":
      return { kind: "literal", value: node.text.slice(1, -1) };
    case "string":
      return { kind: "literal", value: decodeDoubleQuotedText(node.text) };
    case "ansi_c_string":
      return { kind: "literal", value: decodeAnsiCString(node.text) };
    case "concatenation": {
      if (hasUnescapedDynamicPattern(node.text)) {
        return { kind: "dynamic", value: decodeUnquotedShellText(node.text) };
      }
      let value = "";
      let dynamic = false;
      for (const child of namedChildren(node)) {
        const childValue = shellWordValue(child);
        value += childValue.value;
        if (childValue.kind !== "literal") {
          dynamic = true;
        }
      }
      return dynamic ? { kind: "dynamic", value } : { kind: "literal", value };
    }
    default:
      return namedChildren(node).some((child) => shellWordValue(child).kind === "dynamic")
        ? { kind: "dynamic", value: decodeUnquotedShellText(node.text) }
        : { kind: "literal", value: decodeUnquotedShellText(node.text) };
  }
}

function commandNameNode(node: TreeSitterNode): TreeSitterNode | null {
  return (
    node.childForFieldName("name") ??
    namedChildren(node).find((child) => child.type === "command_name") ??
    null
  );
}

function argvFromCommand(node: TreeSitterNode, nameNode: TreeSitterNode): CommandArgv | null {
  if (hasEscapedLineContinuation(nameNode.text) || hasExecutableLineContinuation(node.text)) {
    return null;
  }
  const executable = shellWordValue(nameNode);
  if (executable.kind !== "literal") {
    return null;
  }

  const skipped = new Set<TreeSitterNode>([nameNode, ...namedChildren(nameNode)]);
  const argv = [executable.value];
  const dynamicArguments: DynamicArgument[] = [];
  for (const child of namedChildren(node)) {
    if (
      skipped.has(child) ||
      child.type === "command_name" ||
      child.type === "variable_assignment" ||
      !COMMAND_ARGUMENT_NODE_TYPES.has(child.type)
    ) {
      continue;
    }
    const value = shellWordValue(child);
    if (value.kind === "dynamic") {
      dynamicArguments.push({ index: argv.length, text: child.text, span: spanFromNode(child) });
    }
    argv.push(value.value);
  }
  return { argv, dynamicArguments };
}

function firstShellToken(text: string): string {
  return text.trimStart().match(/^\S+/)?.[0] ?? "";
}

function argvFromDeclarationCommand(node: TreeSitterNode): CommandArgv | null {
  const executable = firstShellToken(node.text);
  if (!executable) {
    return null;
  }
  const argv = [executable];
  const dynamicArguments: DynamicArgument[] = [];
  for (const child of namedChildren(node)) {
    if (!COMMAND_ARGUMENT_NODE_TYPES.has(child.type) && child.type !== "variable_assignment") {
      continue;
    }
    const value = shellWordValue(child);
    if (value.kind === "dynamic") {
      dynamicArguments.push({ index: argv.length, text: child.text, span: spanFromNode(child) });
    }
    argv.push(value.value);
  }
  return { argv, dynamicArguments };
}

function appendTestCommandArguments(
  node: TreeSitterNode,
  argv: string[],
  dynamicArguments: DynamicArgument[],
): void {
  if (node.type === "test_operator" || COMMAND_ARGUMENT_NODE_TYPES.has(node.type)) {
    const value = shellWordValue(node);
    if (value.kind === "dynamic") {
      dynamicArguments.push({ index: argv.length, text: node.text, span: spanFromNode(node) });
    }
    argv.push(value.value);
    return;
  }
  for (const child of namedChildren(node)) {
    appendTestCommandArguments(child, argv, dynamicArguments);
  }
}

function argvFromTestCommand(node: TreeSitterNode): CommandArgv | null {
  const trimmed = node.text.trimStart();
  const executable = trimmed.startsWith("[[") ? "[[" : trimmed.startsWith("[") ? "[" : "";
  if (!executable) {
    return null;
  }
  const argv = [executable];
  const dynamicArguments: DynamicArgument[] = [];
  for (const child of namedChildren(node)) {
    appendTestCommandArguments(child, argv, dynamicArguments);
  }
  return { argv, dynamicArguments };
}

function isCommandLikeNode(node: TreeSitterNode): boolean {
  return (
    node.type === "command" || node.type === "declaration_command" || node.type === "test_command"
  );
}

function recordShape(node: TreeSitterNode, output: MutableExplanation): void {
  if (
    (node.type === "program" || node.type === "list") &&
    (hasDirectChildType(node, ";") || namedChildren(node).filter(isCommandLikeNode).length > 1)
  ) {
    output.shapes.add("sequence");
  }
  if (hasDirectChildType(node, "&")) {
    output.shapes.add("background");
  }
  if (node.type === "pipeline") {
    output.shapes.add("pipeline");
  }
  if (node.type === "list") {
    if (hasDirectChildType(node, "&&")) {
      output.shapes.add("and");
    }
    if (hasDirectChildType(node, "||")) {
      output.shapes.add("or");
    }
  }
  if (node.type === "if_statement") {
    output.shapes.add("if");
  }
  if (node.type === "for_statement") {
    output.shapes.add("for");
  }
  if (node.type === "while_statement") {
    output.shapes.add("while");
  }
  if (node.type === "case_statement") {
    output.shapes.add("case");
  }
  if (node.type === "subshell") {
    output.shapes.add("subshell");
  }
  if (node.type === "compound_statement") {
    output.shapes.add("group");
  }
}

function shellCommandFlag(
  argv: string[],
  startIndex: number,
): { flag: string; index: number } | null {
  const shell = normalizeExecutableToken(argv[startIndex - 1] ?? argv[0] ?? "");
  for (let index = startIndex; index < argv.length; index += 1) {
    const token = argv[index]?.trim();
    if (!token) {
      continue;
    }
    if (token === "--") {
      break;
    }
    const lower = token.toLowerCase();
    if (shell === "cmd") {
      if (lower === "/c" || lower === "/k") {
        return { flag: token, index };
      }
      continue;
    }
    if (shell === "powershell" || shell === "pwsh") {
      if (
        lower === "-c" ||
        lower === "-command" ||
        lower === "--command" ||
        lower === "-encodedcommand" ||
        lower === "-enc" ||
        lower === "-e" ||
        lower === "-f" ||
        lower === "-file"
      ) {
        return { flag: token, index };
      }
      continue;
    }
    if (lower === "-c" || lower === "--command") {
      return { flag: token, index };
    }
    if (token.startsWith("-") && !token.startsWith("--") && lower.slice(1).includes("c")) {
      return { flag: token, index };
    }
  }
  return null;
}

type InlineEvalHit = NonNullable<ReturnType<typeof detectInterpreterInlineEvalArgv>>;

function detectCarrierInlineEvalArgv(argv: string[]): InlineEvalHit | null {
  const dispatchUnwrap = unwrapKnownDispatchWrapperInvocation(argv);
  if (dispatchUnwrap.kind === "unwrapped") {
    return detectInterpreterInlineEvalArgv(dispatchUnwrap.argv);
  }

  const executable = normalizeExecutableToken(argv[0] ?? "");
  if (!SHELL_CARRIER_EXECUTABLES.has(executable)) {
    return null;
  }
  for (let index = 1; index < argv.length; index += 1) {
    const hit = detectInterpreterInlineEvalArgv(argv.slice(index));
    if (hit) {
      return hit;
    }
  }
  return null;
}

function envSplitStringFlag(argv: string[]): string | null {
  if (normalizeExecutableToken(argv[0] ?? "") !== "env") {
    return null;
  }
  for (const arg of argv.slice(1)) {
    const token = arg.trim();
    if (token === "-S" || token === "--split-string") {
      return token;
    }
    if (token.startsWith("--split-string=") || (token.startsWith("-S") && token.length > 2)) {
      return token.startsWith("--") ? "--split-string" : "-S";
    }
  }
  return null;
}

function recordInlineEvalRisk(
  inlineEval: InlineEvalHit,
  text: string,
  span: SourceSpan,
  output: MutableExplanation,
): void {
  output.risks.push({
    kind: "inline-eval",
    command: inlineEval.normalizedExecutable,
    flag: inlineEval.flag,
    text,
    span,
  });
}

function recordDynamicArgumentRisks(
  command: string,
  dynamicArguments: DynamicArgument[],
  output: MutableExplanation,
): void {
  for (const argument of dynamicArguments) {
    output.risks.push({
      kind: "dynamic-argument",
      command,
      argumentIndex: argument.index,
      text: argument.text,
      span: argument.span,
    });
  }
}

function recordCommandRisks(
  argv: string[],
  dynamicArguments: DynamicArgument[],
  text: string,
  span: SourceSpan,
  output: MutableExplanation,
): void {
  const executable = argv[0];
  if (!executable) {
    return;
  }
  const normalizedExecutable = normalizeExecutableToken(executable);
  recordDynamicArgumentRisks(normalizedExecutable, dynamicArguments, output);
  const inlineEval = detectInterpreterInlineEvalArgv(argv) ?? detectCarrierInlineEvalArgv(argv);
  if (inlineEval) {
    recordInlineEvalRisk(inlineEval, text, span, output);
  }

  const shellWrapper = extractShellWrapperCommand(argv);
  if (shellWrapper.isWrapper && shellWrapper.command) {
    const transportArgv = resolveShellWrapperTransportArgv(argv) ?? argv;
    const shellExecutable = transportArgv[0] ?? executable;
    const commandFlag = shellCommandFlag(transportArgv, 1) ?? shellCommandFlag(argv, 1);
    if (isShellWrapperExecutable(executable)) {
      output.risks.push({
        kind: "shell-wrapper",
        executable: shellExecutable,
        flag: commandFlag?.flag ?? "-c",
        payload: shellWrapper.command,
        text,
        span,
      });
    } else {
      output.risks.push({
        kind: "shell-wrapper-through-carrier",
        command: normalizedExecutable,
        text,
        span,
      });
    }
  }

  if (normalizedExecutable === "find") {
    const flag = argv.find((arg) => ["-exec", "-execdir", "-ok", "-okdir"].includes(arg));
    if (flag) {
      output.risks.push({ kind: "command-carrier", command: executable, flag, text, span });
    }
  }
  if (normalizedExecutable === "xargs") {
    output.risks.push({ kind: "command-carrier", command: normalizedExecutable, text, span });
  }
  const splitStringFlag = envSplitStringFlag(argv);
  if (splitStringFlag) {
    output.risks.push({
      kind: "command-carrier",
      command: normalizedExecutable,
      flag: splitStringFlag,
      text,
      span,
    });
  }
  if (normalizedExecutable === "eval") {
    output.risks.push({ kind: "eval", text, span });
  }
  if (SOURCE_EXECUTABLES.has(normalizedExecutable)) {
    output.risks.push({ kind: "source", command: normalizedExecutable, text, span });
  }
  if (normalizedExecutable === "alias") {
    output.risks.push({ kind: "alias", text, span });
  }
  if (!shellWrapper.isWrapper && SHELL_CARRIER_EXECUTABLES.has(normalizedExecutable)) {
    const shellIndex = argv.findIndex((arg) => isShellWrapperExecutable(arg));
    if (shellIndex >= 0 && shellCommandFlag(argv, shellIndex + 1)) {
      output.risks.push({
        kind: "shell-wrapper-through-carrier",
        command: normalizedExecutable,
        text,
        span,
      });
    }

    const carriedCommand = argv.slice(1).find((arg) => {
      const normalized = normalizeExecutableToken(arg);
      return normalized === "eval" || SOURCE_EXECUTABLES.has(normalized);
    });
    const normalizedCarriedCommand = carriedCommand
      ? normalizeExecutableToken(carriedCommand)
      : undefined;
    if (normalizedCarriedCommand === "eval") {
      output.risks.push({ kind: "eval", text, span });
    } else if (normalizedCarriedCommand && SOURCE_EXECUTABLES.has(normalizedCarriedCommand)) {
      output.risks.push({
        kind: "source",
        command: normalizedCarriedCommand,
        text,
        span,
      });
    }
  }
}

function walk(node: TreeSitterNode, output: MutableExplanation, context: CommandContext): void {
  recordShape(node, output);

  const span = spanFromNode(node);
  let childContext = context;
  if (node.type === "program" && hasEscapedLineContinuation(node.text)) {
    output.risks.push({ kind: "line-continuation", text: node.text, span });
  }

  if (node.type === "function_definition") {
    const nameNode = node.childForFieldName("name");
    output.risks.push({
      kind: "function-definition",
      name: nameNode?.text ?? "",
      text: node.text,
      span,
    });
    childContext = "function-definition";
  } else if (node.type === "command_substitution") {
    output.risks.push({ kind: "command-substitution", text: node.text, span });
    childContext = "command-substitution";
  } else if (node.type === "process_substitution") {
    output.risks.push({ kind: "process-substitution", text: node.text, span });
    childContext = "process-substitution";
  } else if (node.type === "heredoc_redirect") {
    output.risks.push({ kind: "heredoc", text: node.text, span });
  } else if (node.type === "herestring_redirect") {
    output.risks.push({ kind: "here-string", text: node.text, span });
  } else if (node.type === "file_redirect") {
    output.risks.push({ kind: "redirect", text: node.text, span });
  } else if (node.type === "ERROR") {
    output.risks.push({ kind: "syntax-error", text: node.text, span });
  }

  if (
    node.type === "command" ||
    node.type === "declaration_command" ||
    node.type === "test_command"
  ) {
    const nameNode = node.type === "command" ? commandNameNode(node) : null;
    const parsed =
      node.type === "command"
        ? nameNode
          ? argvFromCommand(node, nameNode)
          : null
        : node.type === "declaration_command"
          ? argvFromDeclarationCommand(node)
          : argvFromTestCommand(node);
    if (node.type === "command" && nameNode && !parsed) {
      output.risks.push({
        kind: "dynamic-executable",
        text: nameNode.text,
        span: spanFromNode(nameNode),
      });
    } else if (parsed) {
      const step: CommandStep = {
        context,
        executable: parsed.argv[0] ?? "",
        argv: parsed.argv,
        text: node.text,
        span,
      };
      if (step.executable) {
        output.commands.push(step);
        recordCommandRisks(parsed.argv, parsed.dynamicArguments, node.text, span, output);
      }
    }
  }
  for (const child of namedChildren(node)) {
    walk(child, output, childContext);
  }
}

export async function explainShellCommand(source: string): Promise<CommandExplanation> {
  const tree = await parseBashForCommandExplanation(source);
  try {
    const output: MutableExplanation = {
      shapes: new Set(),
      commands: [],
      risks: [],
    };
    walk(tree.rootNode, output, "top-level");
    const topLevelCommands = output.commands.filter((command) => command.context === "top-level");
    return {
      ok: !tree.rootNode.hasError,
      source,
      shapes: [...output.shapes],
      topLevelCommands,
      nestedCommands: output.commands.filter((command) => command.context !== "top-level"),
      risks: output.risks,
    };
  } finally {
    tree.delete();
  }
}
