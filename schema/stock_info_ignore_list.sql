--
-- PostgreSQL database dump
--

-- Dumped from database version 14.7 (Ubuntu 14.7-0ubuntu0.22.04.1)
-- Dumped by pg_dump version 14.7 (Ubuntu 14.7-0ubuntu0.22.04.1)

-- Started on 2023-04-18 10:55:34 CST

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- TOC entry 322 (class 1259 OID 233535)
-- Name: stock_info_ignore_list; Type: TABLE; Schema: public; Owner: capstone
--

CREATE TABLE public.stock_info_ignore_list (
    symbol character varying NOT NULL
);


ALTER TABLE public.stock_info_ignore_list OWNER TO capstone;

--
-- TOC entry 3294 (class 2606 OID 233541)
-- Name: stock_info_ignore_list stock_info_ignore_list_pkey; Type: CONSTRAINT; Schema: public; Owner: capstone
--

ALTER TABLE ONLY public.stock_info_ignore_list
    ADD CONSTRAINT stock_info_ignore_list_pkey PRIMARY KEY (symbol);


-- Completed on 2023-04-18 10:55:34 CST

--
-- PostgreSQL database dump complete
--

